# Tuesday expected output

**Group:** Goal Diggers  
**Activity:** Load and inventory TrashNet image paths and labels in Python/Pandas; inspect formats and duplicates; plan resizing and label encoding.  
**Deliverable:** Dataset inventory and a preprocessing notebook ready for training.

We prepared the creators' resized TrashNet archive. The inventory contains one row for each of 2,527 source images, with its path, label, numeric class identifier, format, dimensions, inclusion status and partition. All source files decoded successfully as RGB JPEG images, each 512 × 384 pixels.

Three pairs had identical decoded pixels but conflicting labels. All six copies were excluded, leaving 2,521 usable images. A perceptual-hash check identified 45 candidate near-duplicate pairs. These require human inspection; connected candidates were kept in the same partition to reduce detected leakage.

The fixed split contains 1,774 training, 371 validation and 376 test images. The manifest records exact membership and its checksum. Training fits models, validation supports selection and test data is reserved for final evaluation after the model choice is frozen.

The label mapping is cardboard = 0, glass = 1, metal = 2, paper = 3, plastic = 4 and trash = 5. These numbers identify categories; they do not measure quantity or rank materials.

Classical models use 64 × 64 grayscale images to produce 1,764 HOG features. Neural models use 160 × 160 RGB images. The small CNN receives pixels scaled to [0, 1]; MobileNetV2 also uses pretrained mean/std normalization. Square resizing changes aspect ratio. Random augmentation applies only to training; validation and test transforms remain deterministic.

The Tuesday notebook demonstrates these steps and exports the inventory, manifest, exclusions, split counts, label mapping, preprocessing plan and review worksheet. It does not train a classifier. The executed copy records a local Windows run, not a verified Google Colab execution.

Human review of labels and candidate duplicates remains pending. Automated outputs do not certify individual member completion.

Source: [TrashNet](https://github.com/garythung/trashnet), supplied weekly plan, and the project's saved preparation reports. AI assisted preparation and execution.
