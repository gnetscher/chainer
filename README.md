# chainer
Utility for describing data and experiment for easily chaining experiments together. A <i>chain</i> is composed by chaining together modules. Each modules consumes something and produces something. 

The convention for creating a module is as following:

```
import chain as ch
#Your module must inherit from class ChainObject
class MyModule(ch.ChainObject):
  #Define the type of objects that are consumed/produced by the module
  _consumer_ = [str]
  _producer_ = [str]
  def __init__(self, prms={}):
    '''
      prms: A dictionary of parameters which influence how what is being
            consumed is converted into produce. For instance if a video is 
            being converted, fps is a parameter. 
    '''
    #(optional) Your custom function that can read default prms and update them 
    #with input parameters
    prms = load_from_defaults(prms) 
    ch.ChainObject(self, prms)
    
    #Handle to the output.
    def produce(self, ip=None):
      pass
  
```




