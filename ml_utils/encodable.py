class Encodable:
  def encode(self):
    raise NotImplementedError('')
  
  @staticmethod
  def add_type(name, klass):
    assert name not in Encodable.typeMapping
    Encodable.typeMapping[name] = klass
  
  @classmethod
  def decode(klass, state):
    if klass is not Encodable:
      raise NotImplementedError('')
    return Encodable.typeMapping[state["$type"]].decode(state)

Encodable.typeMapping = {}
