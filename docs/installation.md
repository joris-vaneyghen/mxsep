## Installation

### From source
```bash
git clone https://github.com/yourusername/music-source-separation
cd music-source-separation
pip install -e .
```

### For data preparation
```bash
pip install -e ".[data]"
```
### For training
```bash
pip install -e ".[train]"
```
### For inference
```bash
pip install -e ".[inference]"
```
### For development
```bash
pip install -e ".[dev]"
```

## Docker Support

```bash
# For training
docker build -f docker/Dockerfile.train -t mxsep-train .
docker run --gpus all -v $(pwd):/workspace mxsep-train

# For inference
docker build -f docker/Dockerfile.inference -t mxsep-inference .
docker run -v $(pwd):/workspace mxsep-inference --model model.pt --input audio.wav
```