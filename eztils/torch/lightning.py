def register_buffer(
    self, name, val
):  # register as same type as weights for lightning modules
    self.register_buffer(name, val.type(self.dtype))
