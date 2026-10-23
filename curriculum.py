"""Resumable HF coverage followed by custom-only fine-tuning."""
import math


class CoverageCurriculum:
    def __init__(self, progress, saved=None, hf_epochs=1.25, custom_epochs=4.,
                 loss_target=1.9, loss_window=100, custom_lr_scale=.25):
        if saved is not None:
            if saved.get('version') != 1:
                raise ValueError('Unsupported curriculum version')
            self.state = dict(saved)
            self.state['recent_custom_losses'] = list(saved['recent_custom_losses'])
        else:
            if (not all(math.isfinite(v) and v > 0 for v in
                        (hf_epochs, custom_epochs, loss_target, custom_lr_scale))
                    or not isinstance(loss_window, int) or loss_window < 1):
                raise ValueError('Invalid curriculum settings')
            self.state = dict(version=1, phase='huggingface',
                hf_target_windows=math.ceil(hf_epochs*progress['huggingface']['windows_per_epoch']),
                custom_required_windows=math.ceil(custom_epochs*progress['custom']['windows_per_epoch']),
                custom_start_windows=None, custom_start_step=None,
                loss_target=loss_target, loss_window=loss_window,
                custom_lr_scale=custom_lr_scale, recent_custom_losses=[])

    @property
    def source(self):
        return self.state['phase']

    def advance(self, progress, completed_step):
        s = self.state
        if s['phase'] == 'huggingface' and progress['huggingface']['windows_traversed'] >= s['hf_target_windows']:
            s.update(phase='custom', custom_start_windows=progress['custom']['windows_traversed'],
                     custom_start_step=completed_step, recent_custom_losses=[])
            return True
        return False

    def record_loss(self, loss):
        if not math.isfinite(loss):
            raise ValueError('Non-finite curriculum loss')
        if self.source == 'custom':
            history = self.state['recent_custom_losses']
            history.append(float(loss))
            del history[:-self.state['loss_window']]

    def complete(self, progress):
        s = self.state
        history = s['recent_custom_losses']
        mean = sum(history)/len(history) if history else float('inf')
        return (self.source == 'custom'
            and progress['custom']['windows_traversed']-s['custom_start_windows'] >= s['custom_required_windows']
            and len(history) >= s['loss_window']
            and mean < s['loss_target']
            and not math.isclose(mean, s['loss_target'], rel_tol=1e-12))

    def lr_factor(self, step):
        if self.source == 'huggingface':
            return 1.
        warmup = min(1., .1+.9*max(0,step-self.state['custom_start_step'])/100)
        return self.state['custom_lr_scale']*warmup

    def report(self, progress):
        s = dict(self.state)
        s.pop('recent_custom_losses')
        history = self.state['recent_custom_losses']
        s['custom_recent_loss'] = sum(history)/len(history) if history else None
        s['custom_loss_samples'] = len(history)
        start = s['custom_start_windows']
        s['custom_phase_windows'] = progress['custom']['windows_traversed']-start if start is not None else 0
        s['custom_phase_percent'] = 100*s['custom_phase_windows']/progress['custom']['windows_per_epoch']
        s['hf_total_percent'] = 100*progress['huggingface']['windows_traversed']/progress['huggingface']['windows_per_epoch']
        return s
