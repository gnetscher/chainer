# chainer
Utility for describing data and experiment for easily chaining experiments together. A <i>chain</i> is composed by chaining together modules. Each modules consumes something and produces something. 

### Creating new modules
The convention for creating a module is at this wiki [page.] (https://github.com/pulkitag/chainer/wiki/Creating-a-New-Module)

###Chaining modules to perform complex operations

Chaining is explained throuth the example of using RCNN for object detection. 
```python
dataSrc  = dc.GetDataDir()
src2Name = imc.DataDir2IterImNames()
name2Im  = imc.File2Im()
bgr      = imc.RGB2BGR()
rcnn     = cc.Im2PersonDet()
imKey    = mc.File2SplitLast()
chain    = ch.Chainer([dataSrc, src2Name, name2Im, bgr,\
           rcnn, (imKey, [(1,0)])], opData=[(-1,0),(-2,1)])
```






