# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import torch


class ResponsesHook(torch.nn.Module):
    """
    A base class that provides a structure to define hooks in PyTorch models.

    The `__call__` method is the main entry point for any custom Hook, and it needs to be implemented by subclasses.
    This method should contain the logic that will trigger when the hooked module's output changes.

    The `update` method is optional and can provide a way to update state or arguments of this hook with new input data
    during runtime, but it's not required for all Hooks.

    """

    def __call__(self, module, input, output):
        raise NotImplementedError

    def register_named_buffers(self, **kwargs) -> None:
        for k, v in kwargs.items():
            self.register_buffer(k, v.to(self.device))

    def update(self, *args, **kwargs):
        """
        Updates the state or arguments of this hook with new input data at runtime.

        This method can be overridden by subclasses to provide custom updating logic. By default, it does nothing and returns None.

        Parameters:
            *args : variable-length argument list
                Variable length argument list that will be used as is for the update operation.

            **kwargs : keyworded arguments
                Keyworded arguments that can also be used to update state or arguments of this hook.

        Returns:
            None
        """
        return None

    def get_thread_handle(self):
        """
        Get thread handle for current instance of class. If no thread has been set or if it does not exist, return None.

        Returns:
            The `thread_handle` attribute of the instance if it exists; otherwise, returns `None`.
        """
        if hasattr(self, "thread_handle"):
            return self.thread_handle
        else:
            return None

    def join(self) -> None:
        """
        Blocks execution of the main thread until all threads started by this instance are done.

        This is useful to ensure that all spawned threads have finished their tasks before the main program continues.
        Without calling `join`, it's possible for your program to exit before all background tasks finish.

        If there are no spawned threads, this method will return immediately.

        Returns:
            None
        """
        if self.get_thread_handle() is not None:
            self.thread_handle.join()
            self.thread_handle = None
