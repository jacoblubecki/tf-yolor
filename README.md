# TensorFlow YOLOR
An attempt at implementing YOLOR in TensorFlow

## Credit

This library is heavily derived from the original repo:
https://github.com/WongKinYiu/yolor

Paper Link: https://arxiv.org/abs/2105.04206

## Disclaimer

No guarantees that this will work properly. I have only tested it on a single
dataset with limited success.

## Quickstart

```bash
pip install git+https://github.com/jacoblubecki/tf-yolor.git
```

And then to create the `YolorP6` model from the paper:
```python
from tfyolor.model import YolorP6

n_classes = 1  # Or however many classes you need to detect.
model = YolorP6(n_classes)
```

## TODO (Coming Soon)

- Loss Function
- Label Utilities
- NMS

## Out of Scope

- Data Loading
- Full Mixed Precision Support
    - This was sort of working previously, but it was not stable.
- Multi-GPU Support
- ONNX Support
- Probably a lot of other things as well...

I don't have a ton of spare time for this project and it was mostly for fun.
This was also my first TensorFlow project, so to be honest there are still a
lot of APIs I haven't gotten a hang of yet, or just don't know about.

These are things I couldn't think of a nice way to port or tried to port, but
ran into difficulties. If there is enough interest, I would consider tackling
some of these. Feel free to open a ticket if you have a suggestion for how to
solve these problems, but please be as clear as possible.

## Contributing

I will consider some pull requests, but I still need to set up some CI stuff
for code style, testing, and documentation. Until that other stuff is setup, I
will probably take a bit to address any PRs so that I can review them more
carefully.
