# How to Export/Import Model Parameters AND Arch to JSON

## Export from Python

1. hybridize your model after it's inited:  

```py
net =  models.init(num_label=num_labels, **arg_dict)

# convert model to Hybrid so it could be exported to JSON
net.hybridize()
```

2. after each epoch, export to JSON:  

```py
# save model parameters AND arch
checkpoint_prefix = os.path.join(prefix, 'hybrid')
net.models[0].export(checkpoint_prefix, epoch=step/1024) # you should use // as floor div in python 3
```

3. you'll get a arch json and multiple param files under working path:  

```shell
hybrid-0000.params
hybrid-0001.params
hybrid-0002.params
...
hybrid-symbol.json
```

## Import to C++

[MXNet Example](https://github.com/apache/incubator-mxnet/blob/master/example/image-classification/predict-cpp/image-classification-predict.cc)

## Reference

[Saving and Loading Gluon Models](https://mxnet.incubator.apache.org/versions/master/tutorials/gluon/save_load_params.html)
