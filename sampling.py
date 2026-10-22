"""Deterministic source-wise epochs with ordered windows and compact resume state."""
import numpy as np


class CoverageSampler:
    def __init__(self, index, context, groups=(), seed=1337):
        self.index, self.context, self.seed = index, context, seed
        self.groups = list(groups) or [dict(name='all', train=[0, len(index)], weight=1.)]
        self.states = [dict(epoch=0, position=0, offset=0, windows=0) for _ in self.groups]
        self.orders = [None] * len(self.groups)
        self.order_epochs = [None] * len(self.groups)
        self.total_windows = []
        for group in self.groups:
            low, count = group['train']
            lengths = index[low:low+count, 1].astype(np.int64)
            self.total_windows.append(int(((lengths-2)//context+1).sum()))

    def next(self, group=0):
        state = self.states[group]
        low, count = self.groups[group]['train']
        if self.order_epochs[group] != state['epoch']:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, group, state['epoch']]))
            self.orders[group] = rng.permutation(count)
            self.order_epochs[group] = state['epoch']
        record = low + int(self.orders[group][state['position']])
        start, length = map(int, self.index[record])
        offset = state['offset']
        size = min(self.context, length-offset-1)
        state['offset'] += size
        state['windows'] += 1
        if state['offset'] == length-1:
            state['offset'] = 0
            state['position'] += 1
            if state['position'] == count:
                state['epoch'] += 1
                state['position'] = 0
        return start+offset, size

    def state_dict(self):
        return dict(version=1, context=self.context, seed=self.seed,
                    groups=[g['train'] for g in self.groups], states=[dict(s) for s in self.states])

    def load_state_dict(self, saved):
        if (saved.get('version') != 1 or saved.get('context') != self.context
                or saved.get('seed') != self.seed or saved.get('groups') != [g['train'] for g in self.groups]
                or len(saved.get('states', [])) != len(self.groups)):
            raise ValueError('Coverage sampler state does not match this corpus')
        for i, state in enumerate(saved['states']):
            if (set(state) != {'epoch','position','offset','windows'}
                    or any(not isinstance(v,int) or v<0 for v in state.values())
                    or state['position'] >= self.groups[i]['train'][1]):
                raise ValueError('Invalid coverage sampler cursor')
            # Recreate the epoch order to verify the offset belongs to this record.
            low, count = self.groups[i]['train']
            order = np.random.default_rng(np.random.SeedSequence([self.seed,i,state['epoch']])).permutation(count)
            length = int(self.index[low+int(order[state['position']]), 1])
            if state['offset'] >= length-1 or state['offset'] % self.context:
                raise ValueError('Invalid coverage window offset')
        self.states = [dict(s) for s in saved['states']]
        self.order_epochs = [None] * len(self.groups)

    def progress(self):
        return {group['name']: dict(completed_epochs=state['epoch'],
                    conversations_in_epoch=state['position'], conversations_per_epoch=group['train'][1],
                    windows_traversed=state['windows'], windows_per_epoch=total,
                    current_epoch_percent=100*(state['windows'] % total)/total)
                for group,state,total in zip(self.groups,self.states,self.total_windows)}


def restarted_lr_factor(step, restart_step, warmup, elapsed, duration):
    """Warm up, hold steady, then decay during the final tenth of the budget."""
    warm = min(1., .1 + .9 * (step-restart_step)/max(1,warmup)) if warmup else 1.
    progress = min(1., max(0., elapsed/max(1e-9,duration)))
    decay = 1. if progress < .9 else 1.-.9*(progress-.9)/.1
    return warm * decay
