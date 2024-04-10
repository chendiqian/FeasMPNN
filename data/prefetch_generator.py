import queue
import threading


class BackgroundGenerator(threading.Thread):
    def __init__(self, dataloader, device, max_prefetch=1):
        threading.Thread.__init__(self)
        self.dataloader = dataloader
        self.queue = queue.Queue(max_prefetch)
        self.device = device
        self.start()

    def __len__(self):
        return len(self.dataloader)

    def run(self):
        for i, batch in enumerate(self.dataloader):
            self.queue.put(batch.to(self.device, non_blocking=True))
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is not None:
            return next_item
        raise StopIteration

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self
