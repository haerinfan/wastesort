# -*- coding: utf-8 -*-
"""Paper content. Inline markup: *italic*, **bold**, [[placeholder]] (yellow highlight)."""

TITLE = "WasteSort: Testing Waste-Material Image Classifiers Beyond TrashNet with Real and Generated Primary Data"

AUTHORS = [
    ("Lordy Allen D. Reyes", "[[e-mail address]]"),
    ("Josh Patrick R. Reyes", "[[e-mail address]]"),
    ("John Benjamin S. Tagbo", "[[e-mail address]]"),
]
AFFILIATION = ["De La Salle University–Dasmariñas", "College of Information and Computer Studies",
               "Dasmariñas, Cavite, Philippines"]

ABSTRACT = (
    "We compare four supervised pipelines for six-class waste-material recognition: histogram-of-oriented-gradients "
    "(HOG) features with a support vector machine (SVM) or a Random Forest, a small convolutional neural network (CNN) "
    "and a frozen-backbone MobileNetV2. They are trained on TrashNet, 2,521 usable photographs of single items on a "
    "white background. On a grouped TrashNet test split of 376 images, MobileNetV2, which was selected on validation "
    "macro F1 before testing, reached 84.57% accuracy and 81.22% macro F1. Trash recall was 55.56% (10 of 18). To test "
    "beyond the benchmark, we built a primary dataset with two parts. The first is real phone photographs the group "
    "took of everyday waste. On the nine single-item photos, MobileNetV2 was correct only 3 times, and the other models "
    "0 to 2 times. The second is 2,521 AI-generated campus scenes, produced with SDXL-Turbo from a fixed prompt plan, "
    "filtered with CLIP and matched to TrashNet's class counts. On these, MobileNetV2 reached [[X.XX%]] macro F1, and "
    "fine-tuning on TrashNet plus generated images gave [[X.XX%]]. Strong benchmark scores did not carry over to real "
    "photographs. The results motivate a larger real-photo collection before any campus use."
)

CCS = [("Computing methodologies", "Supervised learning by classification"),
       ("Computing methodologies", "Computer vision"),
       ("Computing methodologies", "Neural networks")]
KEYWORDS = ("Waste classification; transfer learning; domain shift; real-world evaluation; synthetic data; "
            "TrashNet; MobileNetV2")

# Each section: ("h1"|"h2"|"p"|"bullets"|"table"|"figure"|"pending_figure", payload)
BODY = [
("h1", "INTRODUCTION"),
("p", "Sorting waste correctly starts with recognising what an item is made of. WasteSort studies whether supervised "
      "image classification can identify the dominant material in a photograph of a single waste item. The goal is an "
      "educational prototype for campus recycling guidance. Material recognition is only a first step: whether an item "
      "can be recycled also depends on contamination and local rules, which this study does not model."),
("p", "Public benchmarks such as TrashNet [16] photograph each item on white posterboard. This makes a controlled "
      "starting point, but a campus application would see grass, desks, walkways, bins, mixed lighting and dirty or "
      "crushed items. A high benchmark score therefore says little about the conditions the tool is meant for. "
      "We therefore built a **primary dataset** with two parts. First, the group photographed real waste items with "
      "phone cameras, held in hand or lying where they were found. Second, because a large labelled photo collection "
      "was not feasible in the available time, and with the instructor's approval, we generated 2,521 campus-like "
      "images of the same six materials with an AI image model, matched to TrashNet in size and class mix. The real "
      "photographs test the models on genuine conditions. The generated images allow a large, controlled comparison."),
("p", "We address four research questions:"),
("bullets", [
    "**RQ1.** How do HOG + SVM, HOG + Random Forest, a small CNN and frozen-backbone MobileNetV2 compare on grouped "
    "TrashNet partitions, and how well does the validation-selected model recognise each class?",
    "**RQ2.** How do the primary and secondary datasets differ in class balance, image appearance and visual features?",
    "**RQ3.** How much performance do TrashNet-trained models keep when tested, unchanged, on the group's real "
    "photographs and on the generated images?",
    "**RQ4.** Does adding generated training images change performance on TrashNet, the generated test split "
    "and the real photographs?"]),
("p", "Our contributions are a reproducible benchmark with documented duplicate handling and validation-only model "
      "selection, a real-photo test of the benchmark models, a fully recorded procedure for generating and filtering "
      "campus-like images, and a matched comparison that measures the gap between studio-style photographs and "
      "everyday scenes."),

("h1", "RELATED WORK"),
("p", "TrashNet supplies six material labels for predominantly isolated objects [16]. TACO instead annotates litter in "
      "context for detection and segmentation [10]. WasteSort classifies one item per image and does not locate objects."),
("p", "For feature-based baselines, HOG describes local edge orientation [3]. Support-vector networks [2] and Random "
      "Forests [1] provide margin-based and ensemble decision rules on that representation. For learned "
      "representations, ImageNet [4] pretraining underlies MobileNetV2 [12], an efficient architecture with inverted "
      "residual blocks. Training uses image augmentation [14] and the AdamW optimiser [8]."),
("p", "Generative models can supply training images when real data are scarce. He et al. [6] found that synthetic images "
      "from text-to-image models help in low-data and pretraining settings, while a domain gap to real images remains. "
      "We generate images with SDXL-Turbo, a few-step diffusion model obtained by adversarial diffusion distillation "
      "[13]. We filter them with CLIP, which scores image–text agreement zero-shot [11]. Torralba and Efros [17] showed "
      "that datasets carry recognisable signatures, which motivates our domain-separability test."),
("p", "For evaluation, Sokolova and Lapalme [15] analyse classification measures. We emphasise macro F1 so the small "
      "trash class counts equally. Leakage between partitions can inflate results [7], so related images are kept in "
      "one partition. Neural scores are not calibrated probabilities [5], so the prototype presents them cautiously."),

("h1", "METHODOLOGY"),
("h2", "Research Design"),
("p", "The study is a quantitative, comparative experiment in supervised multiclass image classification. All "
      "candidates use the same records and partitions. Settings were written to protocol files before training "
      "(protocol.json) and before generation (protocol_primary.json). Seed 42 controls random operations. Classical "
      "models use Scikit-learn [9], and neural models use PyTorch and Torchvision."),
("h2", "Secondary Dataset: TrashNet"),
("p", "We downloaded the creators' resized TrashNet archive, 2,527 RGB images at 512 × 384 pixels, from a pinned "
      "revision and verified its checksum. Exact-pixel comparison found three image pairs with contradictory labels. "
      "All six files were excluded, leaving 2,521 usable images (Table 1). A 64-bit perceptual hash with Hamming distance "
      "≤ 6 linked candidate near-duplicates into 2,476 groups. StratifiedGroupKFold with 20 folds assigned folds 0–13 to "
      "training, 14–16 to validation and 17–19 to test. This gives 1,774 training, 371 validation and 376 test images, "
      "with no group crossing partitions."),
("table", "t1"),
("h2", "Primary Dataset: Real Photographs"),
("p", "The group photographed 11 waste items and scenes with phone cameras in November 2026, at home and in public "
      "places. Most items are held in hand against cluttered backgrounds. Labels were assigned by the group from the "
      "physical material, before any model predictions were seen. Nine photos show one item and are scored: plastic "
      "(6: a courier pouch, a pump nozzle, a food tub, a garbage bag, a soft-drink bottle and air-cushion packaging), "
      "paper (2: a shopping bag and a crumpled bag) and metal (1: a food can). Two scenes with many items, littered "
      "bottles on a walkway and mixed waste on pavement, are kept for illustration but not scored, because the "
      "classifier expects one item. Bystanders in one scene were blurred. Exact-pixel checks confirmed that no photo "
      "matches a TrashNet image."),
("h2", "Primary Dataset: Generated Images"),
("p", "The second part is AI-generated. These images are not photographs of real objects, and every file carries "
      "that disclosure in its metadata. Class targets copy TrashNet's usable counts exactly (Table 1), so differences "
      "between the datasets reflect the images rather than sample size or class balance."),
("p", "**Prompt plan.** Each class has 10 item types, for example a green glass bottle, an aluminium soda can or a "
      "styrofoam takeout box. There are 12 campus settings, for example a concrete walkway, a cafeteria table, grass or "
      "a library desk, giving 120 item–setting groups per class. Candidates cycle through every item–setting pair. A "
      "condition (clean, crushed, wet, stained…), a lighting (daylight, overcast, fluorescent, warm, dim) and a camera "
      "angle are drawn with a fixed seed. The template is *“a realistic smartphone photo of {condition} {item}, "
      "{setting}, {lighting}, {angle}, single object centred and in focus.”* Each class uses all 12 settings "
      "about equally. Every image is a separate generation with its own seed, so backgrounds vary within a setting "
      "type, though images of the same type look more alike than images of different types. One setting, a plain "
      "white background, is kept as a TrashNet-like control (about 1/12 of the images)."),
("p", "**Generation.** SDXL-Turbo [13] (revision 71153311) produced 1.8 candidates per target image, 4,541 in total, "
      "with 4 inference steps, guidance 0 and one fixed seed per candidate. Each 512 × 512 output was centre-cropped "
      "to 4:3 and resized to 512 × 384, the TrashNet frame. Both datasets are then distorted identically when resized "
      "to the 160 × 160 network input."),
("p", "**Quality control.** CLIP ViT-B/32 [11] scored each candidate against six class descriptions. A candidate was "
      "kept only if (i) its intended class was CLIP's top choice, (ii) that probability was at least 0.40, and (iii) "
      "it led the runner-up by at least 0.10. Exact pixel duplicates and perceptual-hash near-duplicates (Hamming ≤ 6), "
      "within the set and against every TrashNet image, were removed. The kept images were then selected round-robin "
      "across item types until each class reached its target. Of 4,541 candidates, [[N]] failed the CLIP rule, [[N]] "
      "were near-duplicates and [[N]] were surplus."),
("p", "**Split.** Images were grouped by class, item and setting, then split with the same 20-fold mapping as "
      "TrashNet, giving [[1,765 / 378 / 378]] training, validation and test images. Groups never cross partitions."),
("h2", "Preprocessing and Models"),
("p", "HOG uses 64 × 64 grayscale input, 9 orientations, 8 × 8-pixel cells, 2 × 2-cell blocks and L2-Hys "
      "normalisation, giving 1,764 features. The SVM (RBF kernel, C ∈ {1, 10}, balanced class weights) uses features "
      "standardised on training data. The Random Forest has 300 trees, depth ∈ {unlimited, 20} and balanced class "
      "weights. Neural inputs are 160 × 160 RGB. The small CNN has three convolution–ReLU–max-pool blocks (16/32/64 "
      "channels), 4 × 4 pooling, dropout 0.3 and a six-way linear layer. MobileNetV2 uses ImageNet weights with the "
      "backbone and batch-norm statistics frozen, plus a new dropout-0.2 linear head. Training-only augmentation is a "
      "horizontal flip, rotation within ±10° and brightness/contrast jitter of 0.1."),
("h2", "Training, Selection and Evaluation"),
("p", "Neural models use AdamW (learning rate 10⁻³, weight decay 10⁻⁴), batch size 32 and at most 20 epochs. Training "
      "stops early after 5 epochs without a validation macro-F1 improvement, and the best epoch is restored. The "
      "deployment model is the candidate with the highest validation macro F1. It was frozen before any test "
      "evaluation. We report accuracy, per-class precision, recall and F1, macro F1, confusion matrices, and 95% "
      "group-bootstrap intervals (2,000 resamples) for the TrashNet test results."),
("h2", "Experiments with the Primary Set"),
("p", "**Dataset comparison (RQ2).** For each image we compute brightness, contrast, saturation, colourfulness, edge "
      "density and the fraction of near-white pixels. ImageNet MobileNetV2 features are projected with PCA. A "
      "logistic-regression domain classifier, with 5-fold cross-validation on balanced samples, measures "
      "separability: 50% means the datasets cannot be told apart."),
("p", "**Domain shift (RQ3).** The five frozen TrashNet checkpoints are evaluated without change on the nine scored "
      "real photographs, on all 2,521 generated images and on the generated test split. The real photographs cover "
      "only three classes, so we report per-photo outcomes instead of macro F1."),
("p", "**Fine-tuning (RQ4).** MobileNetV2 is retrained with identical settings on TrashNet training plus generated "
      "training images. Its epoch is chosen on the combined validation split, then it is tested once on each test "
      "split and on the real photographs. The original deployment model is not replaced. The real photographs are "
      "never used for training or model selection."),

("h1", "RESULTS"),
("h2", "TrashNet Benchmark (RQ1)"),
("p", "Table 2 compares the pipelines on the held-out TrashNet test split. MobileNetV2 had the highest validation macro "
      "F1 (84.74%) and was frozen as the deployment model before testing. It achieved 84.57% test accuracy and 81.22% "
      "macro F1, with a bootstrap interval of 76.25–85.60%. The MobileNetV2-minus-CNN macro-F1 difference interval was "
      "+17.25 to +30.21 points. The HOG pipelines and the small CNN trailed by 18 to 24 points of macro F1."),
("table", "t2"),
("p", "Per-class results (Table 3, Figure 1) show the weakness that aggregate scores hide. Cardboard and paper exceeded "
      "89% F1, but trash reached only 58.82% F1, with 10 of 18 trash images recovered. Most errors were between "
      "visually similar materials: glass with metal and plastic, and cardboard with paper. Class-weighted loss raised "
      "small-CNN trash recall from 55.56% to 72.22% (13 of 18), but the interval for the change (−7.69 to +40.00 "
      "points) includes zero."),
("table", "t3"),
("figure", "fig1"),
("h2", "The Group's Real Photographs (RQ3)"),
("p", "The benchmark result did not carry over to real photographs (Table 4). MobileNetV2, 84.57% accurate on TrashNet, "
      "classified only 3 of the 9 scored photos correctly: the food can, the crumpled paper bag and the plastic food "
      "tub. Plastic film and bags, the courier pouch, the garbage bag and the air-cushion packaging, were called "
      "cardboard or paper. The soft-drink bottle was called glass, and the paper shopping bag cardboard. The other "
      "models did no better: weighted CNN 2/9, small CNN 1/9, SVM 1/9 and Random Forest 0/9 [[confirm the four "
      "non-MobileNet counts in the pinned Colab environment]]. With nine photos and three classes, these are case "
      "observations, not an accuracy estimate."),
("table", "t7"),
("h2", "Primary versus Secondary Dataset (RQ2)"),
("p", "[[Fill from reports/primary_set/dataset_comparison.json and dataset_statistics.csv. Expected points to "
      "check: white-background fraction (TrashNet is mostly white posterboard), saturation and edge density "
      "(cluttered scenes), and the domain-separability accuracy.]]"),
("table", "t4"),
("pending_figure", "Figure 2. Random examples per class: TrashNet (left) and generated primary images (right). "
                   "[[Insert reports/primary_set/sample_grid.png]]"),
("h2", "Domain Shift on Generated Images (RQ3)"),
("p", "[[Fill from reports/primary_set/domain_shift.csv: macro F1 of each frozen model on the generated images, the change "
      "from its TrashNet test score, trash recall, and the hardest settings from domain_shift.json "
      "(mobilenet_accuracy_by_setting).]]"),
("table", "t5"),
("h2", "Fine-Tuning with Primary-Set Images (RQ4)"),
("p", "[[Fill from reports/primary_set/finetune_comparison.csv: does adding generated training images raise generated-test "
      "test F1, and does TrashNet test F1 stay level or fall?]]"),
("table", "t6"),

("h1", "DISCUSSION"),
("p", "On TrashNet, transfer learning outperformed both engineered features and a small network trained from scratch. "
      "This is consistent with pretrained representations being valuable when only about 1,800 training images are "
      "available. The comparison does not isolate architecture, because pretraining adds prior information. The trash "
      "class remains the practical bottleneck: it is the smallest class and the most visually varied. With 18 test "
      "images, its recall estimate is uncertain."),
("p", "The real photographs show the gap directly. Hands, cluttered rooms and soft, crumpled plastic never appear in "
      "TrashNet, whose plastic class is mostly rigid bottles and containers on white posterboard. The models seem to "
      "rely on background and shape cues that do not transfer. Brown and dark crumpled film was read as paper or "
      "cardboard, and a clear bottle as glass. [[Interpret RQ2–RQ4 from the generated-image results: whether the "
      "generated scenes reproduce this gap, and whether training on them improves the real-photo outcomes.]]"),
("p", "Two properties of the generated images shape interpretation. First, CLIP filtering keeps images that a pretrained "
      "model already recognises, so the accepted set may be easier than all generated images. This is why rejection "
      "counts are reported. Second, labels come from prompts, not from inspecting physical objects. The generator may "
      "render implausible items or the wrong material, so the contact sheet must be reviewed by people."),

("h1", "LIMITATIONS AND ETHICS"),
("p", "The real photographs are a small convenience sample: 9 scored photos covering three classes, taken by the "
      "authors. They show failure cases but cannot estimate accuracy. The generated images are AI-made and disclosed "
      "as such. Results on them measure transfer to generated scenes, not accuracy on real waste. The study uses one seed, one split and one generator, and diffusion "
      "output can vary slightly across GPUs. SDXL-Turbo is used under a non-commercial research licence, and the set "
      "is used only for this course study. TrashNet is credited to its creators under the MIT licence. No human "
      "participants were involved. Bystanders visible in one photograph were blurred before publication. A material prediction does not certify recyclability, "
      "cleanliness or safety, and the prototype must not replace posted disposal rules."),

("h1", "CONCLUSION"),
("p", "Frozen-backbone MobileNetV2 was the strongest of four pipelines on TrashNet (81.22% macro F1), but it recognised "
      "trash poorly and was correct on only 3 of 9 real photographs taken by the group. A matched set of 2,521 generated "
      "campus-like images makes this gap measurable at scale. [[Summarise the generated-image and fine-tuning outcome "
      "in one sentence.]] A validation set of real campus photographs, with verified disposal rules, remains the necessary "
      "next step."),

("h1", "ACKNOWLEDGMENTS"),
("p", "We thank Gary Thung and Mindy Yang for releasing TrashNet. OpenAI ChatGPT/Codex and Anthropic Claude assisted "
      "with code, data auditing, experiment execution and drafting. The authors reviewed, ran and are responsible for "
      "the reported work."),
]

REFERENCES = [
    "Breiman, L. 2001. Random forests. *Machine Learning* 45, 1, 5–32. DOI: https://doi.org/10.1023/A:1010933404324",
    "Cortes, C. and Vapnik, V. 1995. Support-vector networks. *Machine Learning* 20, 3, 273–297. DOI: https://doi.org/10.1007/BF00994018",
    "Dalal, N. and Triggs, B. 2005. Histograms of oriented gradients for human detection. In *Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR)*, Vol. 1, 886–893.",
    "Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. 2009. ImageNet: A large-scale hierarchical image database. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 248–255.",
    "Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. 2017. On calibration of modern neural networks. In *Proceedings of the 34th International Conference on Machine Learning*, PMLR 70, 1321–1330.",
    "He, R., Sun, S., Yu, X., Xue, C., Zhang, W., Torr, P., Bai, S., and Qi, X. 2023. Is synthetic data from generative models ready for image recognition? In *International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/2210.07574",
    "Kapoor, S. and Narayanan, A. 2023. Leakage and the reproducibility crisis in machine-learning-based science. *Patterns* 4, 9, 100804. DOI: https://doi.org/10.1016/j.patter.2023.100804",
    "Loshchilov, I. and Hutter, F. 2019. Decoupled weight decay regularization. In *International Conference on Learning Representations (ICLR)*. https://arxiv.org/abs/1711.05101",
    "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., and Duchesnay, É. 2011. Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research* 12, 2825–2830.",
    "Proença, P. F. and Simões, P. 2020. TACO: Trash annotations in context for litter detection. arXiv:2003.06975. https://arxiv.org/abs/2003.06975",
    "Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., and Sutskever, I. 2021. Learning transferable visual models from natural language supervision. In *Proceedings of the 38th International Conference on Machine Learning*, PMLR 139, 8748–8763.",
    "Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., and Chen, L.-C. 2018. MobileNetV2: Inverted residuals and linear bottlenecks. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 4510–4520.",
    "Sauer, A., Lorenz, D., Blattmann, A., and Rombach, R. 2023. Adversarial diffusion distillation. arXiv:2311.17042. https://arxiv.org/abs/2311.17042",
    "Shorten, C. and Khoshgoftaar, T. M. 2019. A survey on image data augmentation for deep learning. *Journal of Big Data* 6, Article 60. DOI: https://doi.org/10.1186/s40537-019-0197-0",
    "Sokolova, M. and Lapalme, G. 2009. A systematic analysis of performance measures for classification tasks. *Information Processing & Management* 45, 4, 427–437. DOI: https://doi.org/10.1016/j.ipm.2009.03.002",
    "Thung, G. and Yang, M. 2017. TrashNet. Dataset and source repository. Retrieved November 7, 2026 from https://github.com/garythung/trashnet",
    "Torralba, A. and Efros, A. A. 2011. Unbiased look at dataset bias. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 1521–1528.",
]

P = "[[TBD]]"
TABLES = {
    "t1": ("Table 1. Images per class in each dataset",
           ["Material", "TrashNet (usable)", "Primary: real photos*", "Primary: generated"],
           [1200, 1150, 1250, 1200],
           [["Cardboard", "403", "0", "403"], ["Glass", "498", "0", "498"], ["Metal", "409", "1", "409"],
            ["Paper", "594", "2", "594"], ["Plastic", "480", "6", "480"], ["Trash", "137", "0", "137"],
            ["**Total**", "**2,521**", "**9 (+2)**", "**2,521**"]],
           "*Scored single-item photos; +2 multi-item scenes not scored. TrashNet source archive: 2,527 images."),
    "t2": ("Table 2. TrashNet held-out test results (%; n = 376)",
           ["Method", "Val. macro F1", "Test acc.", "Test macro F1", "Trash recall"],
           [1720, 770, 770, 770, 770],
           [["HOG + SVM", "60.62", "65.96", "63.52", "38.89"],
            ["HOG + Random Forest", "55.94", "63.56", "60.22", "33.33"],
            ["Small CNN", "62.07", "59.57", "57.68", "55.56"],
            ["**MobileNetV2**", "**84.74**", "**84.57**", "**81.22**", "**55.56**"],
            ["Small CNN, weighted*", "60.32", "58.78", "58.84", "72.22"]],
           "*Class-weighting ablation, excluded from deployment selection."),
    "t3": ("Table 3. MobileNetV2 per-class TrashNet test results (%)",
           ["Class", "Precision", "Recall", "F1", "Support"],
           [1200, 900, 900, 900, 900],
           [["Cardboard", "96.30", "88.14", "92.04", "59"], ["Glass", "80.82", "80.82", "80.82", "73"],
            ["Metal", "81.25", "82.54", "81.89", "63"], ["Paper", "86.14", "93.55", "89.69", "93"],
            ["Plastic", "85.29", "82.86", "84.06", "70"], ["Trash", "62.50", "55.56", "58.82", "18"],
            ["**Macro avg.**", "**82.05**", "**80.58**", "**81.22**", "**376**"]]),
    "t4": ("Table 5. Mean image statistics (0–1 scale)",
           ["Measure", "TrashNet", "Primary: generated"],
           [2000, 1400, 1400],
           [["Brightness", P, P], ["Contrast", P, P], ["Saturation", P, P], ["White-background fraction", P, P],
            ["Colourfulness", P, P], ["Edge density", P, P], ["Domain separability (acc.)", P, ""]]),
    "t5": ("Table 6. Frozen TrashNet models tested on the generated images (macro F1, %)",
           ["Method", "TrashNet test", "Generated", "Change", "Gen. trash recall"],
           [1600, 800, 800, 800, 800],
           [["HOG + SVM", "63.52", P, P, P], ["HOG + Random Forest", "60.22", P, P, P],
            ["Small CNN", "57.68", P, P, P], ["MobileNetV2", "81.22", P, P, P],
            ["Small CNN, weighted", "58.84", P, P, P]]),
    "t6": ("Table 7. MobileNetV2 trained on TrashNet only vs TrashNet + generated images (%)",
           ["Trained on", "Tested on", "Acc.", "Macro F1", "Trash recall"],
           [1250, 1150, 800, 800, 800],
           [["TrashNet", "TrashNet test", "84.57", "81.22", "55.56"], ["TrashNet", "Generated test", P, P, P], ["TrashNet", "Real photos (9)", "33.33", "–", "–"],
            ["TrashNet + generated", "TrashNet test", P, P, P], ["TrashNet + generated", "Generated test", P, P, P], ["TrashNet + generated", "Real photos (9)", P, "–", "–"]]),
}

TABLES["t7"] = ("Table 4. MobileNetV2 on the group's real photographs",
           ["Photo", "Item (actual)", "Predicted", "Score"],
           [700, 2000, 1100, 900],
           [["R1", "Courier pouch (plastic)", "Paper", "0.68"], ["R2", "Food can (metal)", "**Metal**", "0.83"],
            ["R3", "Pump nozzle (plastic)", "Cardboard", "0.45"], ["R4", "Shopping bag (paper)", "Cardboard", "0.57"],
            ["R7", "Crumpled bag (paper)", "**Paper**", "0.64"], ["R8", "Food tub (plastic)", "**Plastic**", "0.73"],
            ["R9", "Garbage bag (plastic)", "Cardboard", "0.73"], ["R10", "Drink bottle (plastic)", "Glass", "0.72"],
            ["R11", "Air cushions (plastic)", "Paper", "0.53"], ["**Total**", "", "**3 / 9**", ""]],
           "Bold = correct. Score = top softmax output (uncalibrated).")

FIGURES = {"fig1": ("mobilenet_confusion.png",
                    "Figure 1. MobileNetV2 confusion matrix on the TrashNet test split (rows: actual; columns: predicted).")}
