from neptune import Run
class Logger:
    def __init__(self, run: Run):
        self.run = run
  
    def on_training_start(self):
        print("----------------------------")
        print(" > Starting training...")

    def on_training_stop(self):
        print("> Training finished :)")

        
