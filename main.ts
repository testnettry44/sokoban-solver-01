// Sokoban solver: push-optimal per room, with a whole-level fallback.
//
// Idea: a level is a hallway with independent rooms off it (a box pushed out of
// a room can never come back, so each room is solved on its own and the
// push counts add up). Each room is solved by push-based A*: a state is the
// sorted box cells plus the player's reachable region (named by its lowest
// cell), g is the pushes made, and h is the cheapest assignment of boxes to
// goals under walking distances that ignore boxes (admissible: one push moves
// one box one square, so it can cut that sum by at most one). Dead squares
// (no pull-back path to a goal) and frozen 2x2 blocks are pruned. Everything
// is budgeted by fixed work caps (state/expansion counts), never by the
// clock, so two runs write identical bytes.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Work caps (counts, not time): per-room A* states, per-room BFS retry states,
// whole-level move-A* states.
const ASTAR_CAP = 300000;
const BFS_CAP = 600000;
const MOVE_CAP = 200000;

const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const STEP = ['u', 'd', 'l', 'r'];
const PUSH = ['U', 'D', 'L', 'R'];

type Board = {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
};

type Room = { boxes: number[]; goals: number[]; done: boolean };
type PushSolution = { pushes: number; moves: string; endPlayer: number };

function parseLevel(text: string): Board {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x] ?? '#';
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

/** Walking distances from every source, over open floor only (boxes ignored). */
function distancesFrom(wall: Uint8Array, w: number, sources: number[]): Int32Array {
  const dist = new Int32Array(wall.length).fill(-1);
  const queue = [...sources];
  for (const s of sources) dist[s] = 0;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    for (let k = 0; k < 4; k++) {
      const q = p + DY[k]! * w + DX[k]!;
      if (!wall[q] && dist[q] < 0) {
        dist[q] = dist[p]! + 1;
        queue.push(q);
      }
    }
  }
  return dist;
}

/** Squares a box can stand on and still reach a goal: pull back from the goals. */
function liveSquares(board: Board): Uint8Array {
  const { w, wall, goals } = board;
  const live = new Uint8Array(wall.length);
  const queue = [...goals];
  for (const g of goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi]!;
    for (let k = 0; k < 4; k++) {
      const d = DY[k]! * w + DX[k]!;
      const y = x - d;
      if (!wall[y] && !wall[y - d] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

/**
 * Flood-fill from `from` around walls and boxes, marking `seen` with `stamp`;
 * returns the lowest square reached (the region's canonical name).
 */
function floodLow(
  w: number,
  wall: Uint8Array,
  occupied: Uint8Array,
  from: number,
  seen: Int32Array,
  stamp: number,
  stack: Int32Array,
): number {
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n]!;
    if (p < low) low = p;
    for (let k = 0; k < 4; k++) {
      const q = p + DY[k]! * w + DX[k]!;
      if (!wall[q] && !occupied[q] && seen[q] !== stamp) {
        seen[q] = stamp;
        stack[n++] = q;
      }
    }
  }
  return low;
}

/** The shortest walk from `from` to `to` around walls and boxes (lowercase LURD). */
function walkTo(
  wall: Uint8Array,
  w: number,
  occupied: Uint8Array,
  from: number,
  to: number,
): string {
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array<string>(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    for (let k = 0; k < 4; k++) {
      const q = p + DY[k]! * w + DX[k]!;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = STEP[k]!;
      queue.push(q);
    }
  }
  if (prev[to] === -1) throw new Error('push square out of reach');
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

/** True when the box just pushed to `t` sits in a frozen 2x2 block. */
function frozenAt(
  t: number,
  occupied: Uint8Array,
  wall: Uint8Array,
  w: number,
  goalSet: Uint8Array,
  size: number,
): boolean {
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    if (a < 0 || a + w + 1 >= size) continue;
    const cells = [a, a + 1, a + w, a + w + 1];
    let filled = true;
    for (const c of cells) {
      if (!wall[c] && !occupied[c]) {
        filled = false;
        break;
      }
    }
    if (!filled) continue;
    for (const c of cells) {
      if (occupied[c] && !goalSet[c]) return true;
    }
  }
  return false;
}

function allPermutations(n: number): number[][] {
  const out: number[][] = [];
  const cur: number[] = [];
  const used = new Array<boolean>(n).fill(false);
  const rec = (): void => {
    if (cur.length === n) {
      out.push([...cur]);
      return;
    }
    for (let i = 0; i < n; i++) {
      if (used[i]) continue;
      used[i] = true;
      cur.push(i);
      rec();
      cur.pop();
      used[i] = false;
    }
  };
  rec();
  return out;
}

// --- binary heap on [f, g, seq, state], smallest f first, then largest g, then earliest ---
function heapLess(a: number[], b: number[]): boolean {
  return (
    a[0]! < b[0]! ||
    (a[0] === b[0] && (a[1]! > b[1]! || (a[1] === b[1] && a[2]! < b[2]!)))
  );
}
function heapPush(heap: number[][], e: number[]): void {
  let i = heap.length;
  heap.push(e);
  while (i > 0) {
    const p = (i - 1) >> 1;
    if (!heapLess(heap[i]!, heap[p]!)) break;
    const tmp = heap[i]!;
    heap[i] = heap[p]!;
    heap[p] = tmp;
    i = p;
  }
}
function heapPop(heap: number[][]): number[] {
  const top = heap[0]!;
  const last = heap.pop()!;
  if (heap.length > 0) {
    heap[0] = last;
    for (let i = 0; ; ) {
      let c = 2 * i + 1;
      if (c >= heap.length) break;
      if (c + 1 < heap.length && heapLess(heap[c + 1]!, heap[c]!)) c++;
      if (!heapLess(heap[c]!, heap[i]!)) break;
      const tmp = heap[i]!;
      heap[i] = heap[c]!;
      heap[c] = tmp;
      i = c;
    }
  }
  return top;
}

/**
 * Push-optimal search over one board (a room, or the whole level when the
 * level does not split into rooms). A* when `useAStar`, plain push-BFS
 * otherwise. Returns null when the state cap is hit or no solution exists.
 */
function solvePush(board: Board, maxStates: number, useAStar: boolean): PushSolution | null {
  const { w, wall, goals, boxes: startBoxes, player: startPlayer } = board;
  const size = wall.length;
  const n = goals.length;
  const goalSet = new Uint8Array(size);
  for (const g of goals) goalSet[g] = 1;
  const live = liveSquares(board);
  if (startBoxes.some((b) => !live[b])) return null;
  if (startBoxes.every((b) => goalSet[b])) {
    return { pushes: 0, moves: '', endPlayer: startPlayer };
  }

  const distPerGoal = goals.map((g) => distancesFrom(wall, w, [g]));
  const perms = useAStar ? allPermutations(n) : [];
  const hCache = new Map<number | string, number>();
  const boxKey = (bs: number[]): number | string => {
    if (n <= 4) {
      let k = 0;
      for (const b of bs) k = k * size + b;
      return k;
    }
    return bs.join(',');
  };
  const heuristic = (bs: number[]): number => {
    const ck = boxKey(bs);
    const cached = hCache.get(ck);
    if (cached !== undefined) return cached;
    let best = Infinity;
    for (const perm of perms) {
      let sum = 0;
      for (let i = 0; i < n; i++) {
        const d = distPerGoal[perm[i]!]![bs[i]!]!;
        if (d < 0) {
          sum = Infinity;
          break;
        }
        sum += d;
      }
      if (sum < best) best = sum;
    }
    hCache.set(ck, best);
    return best;
  };
  const stateKey = (bs: number[], low: number): number | string => {
    if (n <= 4) {
      let k = low;
      for (const b of bs) k = k * size + b;
      return k;
    }
    return low + ':' + bs.join(',');
  };

  const allBoxes: number[][] = [startBoxes.slice()];
  const allPlayer: number[] = [startPlayer];
  const allKey: (number | string)[] = [];
  const parent: number[] = [-1];
  const fromCell: number[] = [-1];
  const toCell: number[] = [-1];
  const dirOf: number[] = [-1];
  const gVal: number[] = [0];
  const best = new Map<number | string, number>();
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stampSeen = 0;
  let stampMark = 0;
  const occupied = new Uint8Array(size);
  for (const b of startBoxes) occupied[b] = 1;
  const startLow = floodLow(w, wall, occupied, startPlayer, seen, ++stampSeen, stack);
  for (const b of startBoxes) occupied[b] = 0;
  allKey.push(stateKey(startBoxes, startLow));
  best.set(allKey[0]!, 0);

  const heap: number[][] = [];
  const queue: number[] = [0];
  let qhead = 0;
  let seq = 0;
  if (useAStar) heapPush(heap, [heuristic(startBoxes), 0, seq++, 0]);

  let expanded = 0;
  while (useAStar ? heap.length > 0 : qhead < queue.length) {
    const idx = useAStar ? heapPop(heap)[3]! : queue[qhead++]!;
    const g = gVal[idx]!;
    if (best.get(allKey[idx]!) !== g) continue; // stale duplicate
    const boxes = allBoxes[idx]!;
    if (boxes.every((b) => goalSet[b])) {
      // Rebuild the moves: walk to each push square, then push.
      const path: number[] = [];
      for (let s = idx; s > 0; s = parent[s]!) path.push(s);
      path.reverse();
      let moves = '';
      let player = startPlayer;
      let curBoxes = startBoxes.slice();
      for (const s of path) {
        const from = fromCell[s]!;
        const to = toCell[s]!;
        const k = dirOf[s]!;
        const d = DY[k]! * w + DX[k]!;
        for (const b of curBoxes) occupied[b] = 1;
        moves += walkTo(wall, w, occupied, player, from - d);
        for (const b of curBoxes) occupied[b] = 0;
        moves += PUSH[k]!;
        player = from;
        curBoxes = allBoxes[s]!;
      }
      return { pushes: path.length, moves, endPlayer: player };
    }
    if (expanded >= maxStates) return null;
    expanded++;
    const player = allPlayer[idx]!;
    for (const b of boxes) occupied[b] = 1;
    floodLow(w, wall, occupied, player, seen, ++stampSeen, stack);
    const here = stampSeen;
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi]!;
      for (let k = 0; k < 4; k++) {
        const d = DY[k]! * w + DX[k]!;
        const t = b + d;
        if (seen[b - d] !== here || wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        if (frozenAt(t, occupied, wall, w, goalSet, size)) {
          occupied[t] = 0;
          occupied[b] = 1;
          continue;
        }
        const nextBoxes = boxes.slice();
        nextBoxes[bi] = t;
        nextBoxes.sort((a, c) => a - c);
        const h = useAStar ? heuristic(nextBoxes) : 0;
        if (h === Infinity) {
          occupied[t] = 0;
          occupied[b] = 1;
          continue;
        }
        const low2 = floodLow(w, wall, occupied, b, mark, ++stampMark, stack);
        const key = stateKey(nextBoxes, low2);
        occupied[t] = 0;
        occupied[b] = 1;
        if (best.has(key) && best.get(key)! <= g + 1) continue;
        if (allBoxes.length >= maxStates) {
          for (const x of boxes) occupied[x] = 0;
          return null;
        }
        best.set(key, g + 1);
        const ni = allBoxes.length;
        allBoxes.push(nextBoxes);
        allPlayer.push(b);
        allKey.push(key);
        parent.push(idx);
        fromCell.push(b);
        toCell.push(t);
        dirOf.push(k);
        gVal.push(g + 1);
        if (useAStar) heapPush(heap, [g + 1 + h, g + 1, seq++, ni]);
        else queue.push(ni);
      }
    }
    for (const x of boxes) occupied[x] = 0;
  }
  return null;
}

/**
 * Split a level into rooms: cut the dead squares of the player's row (the
 * hallway) out of the floor; each remaining component holding boxes is a
 * room. Null when the level does not split cleanly.
 */
function roomsOf(board: Board): Room[] | null {
  const { w, wall, goals, boxes, player } = board;
  const live = liveSquares(board);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) {
    if (!wall[i] && !live[i]) cut[i] = 1;
  }
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (let k = 0; k < 4; k++) {
        const q = squares[i]! + DY[k]! * w + DX[k]!;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ boxes: [], goals: [], done: false });
  }
  for (const b of boxes) {
    if (part[b]! < 0) return null;
    rooms[part[b]!]!.boxes.push(b);
  }
  for (const g of goals) {
    if (part[g]! < 0) return null;
    rooms[part[g]!]!.goals.push(g);
  }
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/** Replay moves on the board; true only when every box ends on a goal. */
function replayOk(board: Board, moves: string): boolean {
  const { w, wall, goals } = board;
  const occupied = new Uint8Array(wall.length);
  for (const b of board.boxes) occupied[b] = 1;
  let player = board.player;
  for (const ch of moves.toLowerCase()) {
    const k = STEP.indexOf(ch);
    if (k < 0) return false;
    const d = DY[k]! * w + DX[k]!;
    const next = player + d;
    if (wall[next]) return false;
    if (occupied[next]) {
      const beyond = next + d;
      if (wall[beyond] || occupied[beyond]) return false;
      occupied[next] = 0;
      occupied[beyond] = 1;
    }
    player = next;
  }
  return goals.every((g) => occupied[g]);
}

/** Whole-level move-based A* (last resort); returns moves or null. */
function solveMoves(board: Board, maxStates: number): string | null {
  const { w, wall, goals, boxes: startBoxes, player: startPlayer } = board;
  const size = wall.length;
  const goalSet = new Uint8Array(size);
  for (const g of goals) goalSet[g] = 1;
  const live = liveSquares(board);
  const near = distancesFrom(wall, w, goals);
  const hOf = (bs: number[]): number => {
    let sum = 0;
    for (const b of bs) {
      const d = near[b]!;
      if (d < 0) return Infinity;
      sum += d;
    }
    return sum;
  };
  const keyOf = (p: number, bs: number[]): string =>
    String.fromCharCode(p, ...bs);
  const allPlayer: number[] = [startPlayer];
  const allBoxes: number[][] = [startBoxes.slice()];
  const parent: number[] = [-1];
  const stepOf: string[] = [''];
  const gVal: number[] = [0];
  const best = new Map<string, number>([[keyOf(startPlayer, startBoxes), 0]]);
  const heap: number[][] = [];
  let seq = 0;
  heapPush(heap, [hOf(startBoxes), 0, seq++, 0]);
  let expanded = 0;
  const occupied = new Uint8Array(size);
  while (heap.length > 0) {
    const idx = heapPop(heap)[3]!;
    const g = gVal[idx]!;
    const boxes = allBoxes[idx]!;
    const player = allPlayer[idx]!;
    if (best.get(keyOf(player, boxes)) !== g) continue;
    if (boxes.every((b) => goalSet[b])) {
      let moves = '';
      for (let s = idx; s > 0; s = parent[s]!) moves = stepOf[s]! + moves;
      return moves;
    }
    if (expanded >= maxStates) return null;
    expanded++;
    for (const b of boxes) occupied[b] = 1;
    for (let k = 0; k < 4; k++) {
      const d = DY[k]! * w + DX[k]!;
      const q = player + d;
      if (wall[q]) continue;
      let nb = boxes;
      let letter = STEP[k]!;
      const at = boxes.indexOf(q);
      if (at >= 0) {
        if (wall[q + d] || occupied[q + d] || !live[q + d]) continue;
        occupied[q] = 0;
        occupied[q + d] = 1;
        const frozen = frozenAt(q + d, occupied, wall, w, goalSet, size);
        occupied[q + d] = 0;
        occupied[q] = 1;
        if (frozen) continue;
        nb = boxes.slice();
        nb[at] = q + d;
        nb.sort((a, c) => a - c);
        letter = PUSH[k]!;
      }
      const key = keyOf(q, nb);
      if (best.has(key) && best.get(key)! <= g + 1) continue;
      if (allBoxes.length >= maxStates) {
        for (const x of boxes) occupied[x] = 0;
        return null;
      }
      best.set(key, g + 1);
      const ni = allBoxes.length;
      allBoxes.push(nb);
      allPlayer.push(q);
      parent.push(idx);
      stepOf.push(letter);
      gVal.push(g + 1);
      heapPush(heap, [g + 1 + hOf(nb), g + 1, seq++, ni]);
    }
    for (const x of boxes) occupied[x] = 0;
  }
  return null;
}

function wholeFallback(board: Board): string {
  const moves = solveMoves(board, MOVE_CAP);
  if (moves !== null && replayOk(board, moves)) return moves;
  return '';
}

function solveLevel(text: string): string {
  try {
    const board = parseLevel(text);
    const rooms = roomsOf(board);
    if (rooms !== null) {
      let player = board.player;
      let moves = '';
      for (const room of rooms) {
        // Other rooms' boxes stand as walls (on their goals once solved).
        const wall2 = board.wall.slice();
        for (const other of rooms) {
          if (other === room) continue;
          for (const b of other.done ? other.goals : other.boxes) wall2[b] = 1;
        }
        const sub: Board = {
          w: board.w,
          h: board.h,
          wall: wall2,
          goals: room.goals.slice().sort((a, b) => a - b),
          boxes: room.boxes.slice().sort((a, b) => a - b),
          player,
        };
        let sol = solvePush(sub, ASTAR_CAP, true);
        if (sol === null) sol = solvePush(sub, BFS_CAP, false);
        if (sol === null) return wholeFallback(board);
        moves += sol.moves;
        player = sol.endPlayer;
        room.done = true;
      }
      if (replayOk(board, moves)) return moves;
      return wholeFallback(board);
    }
    let sol = solvePush(board, ASTAR_CAP, true);
    if (sol === null) sol = solvePush(board, BFS_CAP, false);
    if (sol !== null) return sol.moves;
    return wholeFallback(board);
  } catch {
    return '';
  }
}

for (const id of readdirSync(INPUT_DIR).sort()) {
  let moves = '';
  try {
    moves = solveLevel(readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8'));
  } catch {
    moves = '';
  }
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), `${moves}\n`);
}
