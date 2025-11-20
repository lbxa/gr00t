# Camera Configuration Guide for GR00T

This guide explains how camera configurations work across LeRobot and GR00T, and how to ensure consistency.

## Overview

When training GR00T with LeRobot, cameras are handled at two levels:

1. **LeRobot Dataset** - Defines cameras as `observation.images.<camera_name>`
2. **GR00T Data Config** - Maps cameras to `video.<camera_name>` for processing

## Your Setup: Front + Top Cameras

Your dataset has:
- `observation.images.front`
- `observation.images.top`

### ✅ Changes Made

#### 1. Created New GR00T Data Config

**File:** `gr00t/gr00t/experiment/data_config.py`

```python
class So100FrontTopCamDataConfig(So100DataConfig):
    """SO-100/SO-101 configuration with front and top cameras (no wrist camera)."""
    video_keys = ["video.front", "video.top"]
    state_keys = ["state.single_arm", "state.gripper"]
    action_keys = ["action.single_arm", "action.gripper"]
    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))
```

Registered as: `"so100_fronttop"` in `DATA_CONFIG_MAP`

#### 2. Created Modality Mapping File

**File:** `gr00t/examples/SO-100/so100_fronttop__modality.json`

```json
{
    "video": {
        "front": {
            "original_key": "observation.images.front"
        },
        "top": {
            "original_key": "observation.images.top"
        }
    }
}
```

This maps:
- `video.front` → `observation.images.front`
- `video.top` → `observation.images.top`

#### 3. Updated Modal Inference Service

**File:** `gr00t/scripts/modal_inference.py`

Changed default config to: `DATA_CONFIG = "so100_fronttop"`

Updated test observation format to use `video.front` and `video.top`.

## How It Works

### During LeRobot Training

1. LeRobot automatically detects ALL cameras in your dataset:
   ```python
   # From lerobot/src/lerobot/policies/groot/processor_groot.py:271
   img_keys = sorted([k for k in obs if k.startswith("observation.images.")])
   ```

2. Your dataset has `observation.images.front` and `observation.images.top`, so both are used.

3. LeRobot stacks them and passes to GR00T as multi-view input.

### During GR00T Inference

1. GR00T expects observations with keys matching the data config:
   ```python
   obs = {
       "video.front": ...,  # Front camera
       "video.top": ...,    # Top camera
       "state.single_arm": ...,
       "state.gripper": ...,
   }
   ```

2. The modality config maps these to the dataset format if needed.

3. GR00T processes both camera views together.

## Usage

### Training (LeRobot + GR00T)

No special configuration needed! LeRobot automatically uses all cameras in your dataset.

### Inference (Modal Service)

```bash
# Deploy with front+top camera config
export DATA_CONFIG="so100_fronttop"
modal deploy gr00t/scripts/modal_inference.py
```

### Making Predictions

```python
import numpy as np

observation = {
    "video.front": np.random.randint(0, 255, (1, 480, 640, 3), dtype=np.uint8),
    "video.top": np.random.randint(0, 255, (1, 480, 640, 3), dtype=np.uint8),
    "state.single_arm": np.array([[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]], dtype=np.float32),
    "state.gripper": np.array([[0.5]], dtype=np.float32),
    "annotation.human.task_description": ["pick up the object"],
}
```

## Available Camera Configurations

| Config Name | Video Keys | Use Case |
|------------|------------|----------|
| `so100` | `["video.webcam"]` | Single camera SO-100/SO-101 |
| `so100_dualcam` | `["video.front", "video.wrist"]` | Front + wrist cameras |
| `so100_fronttop` | `["video.front", "video.top"]` | Front + top cameras (your setup) ✨ |

## Creating Custom Camera Configs

If you need a different camera setup:

### 1. Create Data Config

```python
# In gr00t/gr00t/experiment/data_config.py
class MyCustomCamConfig(So100DataConfig):
    video_keys = ["video.cam1", "video.cam2", "video.cam3"]  # Your cameras
    # ... rest stays the same
```

### 2. Register It

```python
DATA_CONFIG_MAP = {
    # ... existing configs
    "my_custom_cam": MyCustomCamConfig(),
}
```

### 3. Create Modality File

```json
{
    "video": {
        "cam1": {
            "original_key": "observation.images.cam1"
        },
        "cam2": {
            "original_key": "observation.images.cam2"
        },
        "cam3": {
            "original_key": "observation.images.cam3"
        }
    }
}
```

### 4. Use It

```bash
export DATA_CONFIG="my_custom_cam"
modal deploy gr00t/scripts/modal_inference.py
```

## Verification Checklist

✅ Dataset has cameras: `observation.images.front`, `observation.images.top`
✅ Data config created: `So100FrontTopCamDataConfig` with `video_keys = ["video.front", "video.top"]`
✅ Modality file created: `so100_fronttop__modality.json` with proper mapping
✅ Config registered: `"so100_fronttop"` in `DATA_CONFIG_MAP`
✅ Modal inference updated: Default config set to `so100_fronttop`
✅ Test scripts updated: Use `video.front` and `video.top`

## Troubleshooting

### Camera not found during inference

**Error:** `KeyError: 'video.wrist'`

**Cause:** Using wrong data config (e.g., `so100_dualcam` expects wrist camera)

**Solution:** Set correct config:
```bash
export DATA_CONFIG="so100_fronttop"
```

### Cameras detected during training but not inference

**Cause:** Mismatch between dataset camera names and data config video_keys

**Solution:** Check your dataset's `info.json` for actual camera names:
```bash
cat path/to/dataset/meta/info.json | grep "observation.images"
```

Then ensure data config matches those names.

## Summary

✅ **LeRobot Training**: Automatically uses all cameras in your dataset
✅ **GR00T Inference**: Uses `so100_fronttop` config for front+top cameras
✅ **Consistency**: Modality file maps between LeRobot and GR00T formats
✅ **Ready to use**: All files created and configured!

