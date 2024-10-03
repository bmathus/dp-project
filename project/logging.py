from neptune import Run
class Logger:
    def __init__(self, run: Run):
        self.run = run
  
    def on_training_start(self):
        print("Training started")

    def on_training_stop(self):
        print("Training finished")

    def on_epoch_complete(self, epoch, stats):
        self.run["train/loss"].append(stats["loss_train"])
        self.run["valid/loss"].append(stats["loss_val"])
        self.run["epoch"].append(epoch)
        print(f"Epoch complete: {epoch}  {stats}")
        

class Statistics:
    def __init__(self):
        self.values = dict()

    def _step(self, key, value):
        sum, count = 0.0, 0.0
        if key in self.values:
            sum, count = self.values[key]
        sum += value
        count += 1.0
        self.values[key] = (sum, count)
    
    def get_averages(self):
        result = dict()
        for k, (sum,count) in self.values.items():
            result[k] = float(sum/count)
        return result
  
    @staticmethod
    def merge(s1, s2):
        result = s1.get_averages()
        result.update(s2.get_averages())
        return result
    
    def train_step(self,loss):
        self._step("loss_train", loss.item())

    def valid_step(self,loss):
        self._step("loss_val", loss.item())

  