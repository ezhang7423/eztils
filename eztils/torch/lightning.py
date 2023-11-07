def register_buffer(self, name, val):
    """Registers a buffer as the same type as weights for lightning modules.

    :param name: The name of the buffer.
    :type name: str
    :param val: The value of the buffer.
    :type val: torch.Tensor
    """
    self.register_buffer(name, val.type(self.dtype))
