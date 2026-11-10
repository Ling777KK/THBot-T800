"""Replay real observation intervals in random contiguous blocks.

One shared PC clock per vector batch; lanes have independent delayed commands.
The clock has its own seeded RNG, never depends on policy actions or deaths.
"""
import json
import math
import random
from collections import Counter
from pathlib import Path


class ReplayClock:
    def __init__(self, source):
        data = json.loads(Path(source).read_text(encoding='utf-8-sig')) if isinstance(source, (str, Path)) else source
        self.rows = data['records']
        self.block = int(data.get('block_decisions', 64))
        self.rows = [dict(r) for r in self.rows]
        self.starts = [i for i in range(len(self.rows)-self.block+1)
                       if all(self.rows[j]['segment']==self.rows[i]['segment']
                              for j in range(i, i+self.block))]
        if not self.starts or self.block < 32:
            raise ValueError('Timing replay needs contiguous blocks of at least 32 decisions')
        for r in self.rows:
            dt = float(r['interval_frames'])
            w = r['delay_weights']
            if not math.isfinite(dt) or not 0 < dt <= 8:
                raise ValueError('Replay interval must be positive and at most eight game frames')
            if not w or any(not math.isfinite(v) or v<0 for v in w) or sum(w)<=0:
                raise ValueError('Replay delay weights are invalid')
        # Bounds the number of unacknowledged commands kept in the 64-slot queue,
        # including a join between two independent replay blocks.
        for start in self.starts:
            for j in range(start, start+self.block-15):
                if sum(r['interval_frames'] for r in self.rows[j:j+16]) <= 9:
                    raise ValueError('Timing replay too dense for bounded command queue')
        self.reset(0)

    def reset(self, seed):
        self.rng = random.Random(int(seed))
        self.phase = self.rng.random()
        self.remaining = 0
        self.index = 0
        self.histogram = Counter()
        self.samples = 0
        self.elapsed_frames = 0

    def next(self):
        if self.remaining == 0:
            self.index = self.rng.choice(self.starts)
            self.remaining = self.block
        row_index = self.index
        row = self.rows[row_index]
        self.index += 1
        self.remaining -= 1
        dt = row['interval_frames']
        end = self.phase + dt
        frames = math.floor(end)
        self.phase = end-frames
        self.samples += 1
        self.elapsed_frames += frames
        self.histogram[frames] += 1
        return row_index, frames

    def summary(self):
        return dict(samples=self.samples, game_frames=self.elapsed_frames,
                    decisions_per_second=60*self.samples/max(1,self.elapsed_frames),
                    interval_histogram=dict(self.histogram), shared_batch_clock=True)
