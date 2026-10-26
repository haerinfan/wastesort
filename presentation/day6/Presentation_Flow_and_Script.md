# Goal Diggers - Presentation Guide and Flow

## 1. Choose the presentation and rehearse

For the full project, use MAIN_PRESENTATION/WasteSort_Saturday_Demo.pptx. Allow 8-10 minutes, including a short app demo. Speaker notes are included; the PDF is your backup.

For ONLY Monday's assigned concepts, use MONDAY_ONLY/WasteSort_Monday_Concepts.pptx and its speaking guide instead (about four minutes). For Tuesday's dataset lesson, use TUESDAY_ONLY and its guide. The weekly guide explains the remaining daily activities.

Suggested roles: Presenter 1 covers slides 1-2; Presenter 2 covers slides 3-4; Presenter 3 covers slide 5, the demo and closing. This is a rehearsal plan, not a claim about who did the work. One person can deliver everything.

Flow: problem -> dataset -> models -> results -> errors -> app demo -> limitations -> questions.

Before class: unzip the pack, open the slides and PDF, read this script, start the app and test one photo. First-time setup needs internet and can take several minutes. Do not train models or install packages during the presentation. The presentation pack does not include the entire application; download the complete project separately from GitHub.

## 2. Script for slides 1-3

SLIDE 1 - WasteSort: from dataset to prediction (45 seconds)

Good day. We are Goal Diggers, and our project is WasteSort. It is a research prototype that predicts the material in one item photograph: cardboard, glass, metal, paper, plastic or trash. It uses supervised learning because the training images have known labels. We prepared the data, compared models and built an application using the selected model. The app predicts a material; campus bin instructions still need verified local rules.

Transition: First, we needed labeled data and a fair way to separate learning from evaluation.

SLIDE 2 - Prepare the data before training (60 seconds)

We used TrashNet, with 2,527 source images. Six files had identical pixels but conflicting labels, so both copies of those pairs were excluded. This left 2,521 usable images: 1,774 for training, 371 for validation and 376 for testing. Training fits the models, validation helps choose settings and the model, and testing evaluates the frozen choice. Similar-image groups stay within one partition to reduce detected duplicate leakage. All 45 flagged pairs have AI-assisted review notes; this is not a complete independent human audit.

Transition: With the data prepared, we compared different ways to recognize materials.

SLIDE 3 - Compare four primary approaches (60 seconds)

Our main comparison includes SVM, Random Forest, a small CNN and MobileNetV2. SVM and Random Forest use HOG features, which summarize edge directions. Neural models use resized RGB images. MobileNetV2 starts from pretrained image features, which is transfer learning. We also completed KNN and Decision Tree exercises and a separate Linear Regression exercise. Regression predicts a numerical target; it is not the waste classifier.

Handoff: Our next presenter will explain the measured results.

## 3. Script for slides 4-6 and closing

SLIDE 4 - MobileNetV2 led the published comparison (75 seconds)

We chose the deployment model using validation results before test evaluation. In the original experiment, MobileNetV2 achieved 84.57 percent test accuracy and 81.22 percent macro F1. Accuracy measures the proportion of correct predictions. Macro F1 weights each class equally, which matters when class sizes differ. These scores describe held-out TrashNet images, not campus photos. A separate Colab run is recorded; some CNN scores differ, so that run is kept separate from this original table.

Transition: Overall accuracy is useful, but the errors reveal a weakness.

SLIDE 5 - Overall accuracy hides missed trash items (60 seconds)

The confusion matrix has actual labels in rows and predictions in columns. Correct predictions lie on the diagonal. MobileNetV2 recognized only 10 of 18 trash images: 55.56 percent recall. It missed eight. This shows why overall accuracy alone is insufficient. There are also few trash test examples, so this is not a reliable estimate of campus performance.

Transition: Now we will show how the model works inside the app.

SLIDE 6 - Demonstrate the app and explain its limits (2 minutes)

The app loads a saved trained model. We upload one clear item photo, select Identify material and read the category. The displayed score is a model score, not a guarantee of correctness. We can download the result and inspect the research-results page. If this is a TrashNet image, it demonstrates the workflow; it is not a new independent evaluation.

Closing: WasteSort connects labeled data, preprocessing, training, evaluation and a working interface. It is a research prototype with recorded limitations. Next steps include instructor review of the methodology changes, verified campus disposal rules and separate campus-photo evaluation. Thank you. We are ready for questions.

## 4. Demo steps and backup plan

1. Download the complete project from https://github.com/haerinfan/wastesort using Code > Download ZIP, then extract it.

2. On Windows, double-click Start WasteSort.cmd. Keep its terminal open and visit the local address it displays. Follow README.md if it reports a missing Python installation. Complete setup before class.

3. Choose a clear JPEG or PNG with one item. Use a dataset photo if available or your own non-sensitive photo. Do not assume its prediction will be correct.

4. Upload the photo and select Identify material. Explain the label and score. If it is wrong, say: This is a model error, which is why we evaluate many examples and report limitations.

5. Download the result. Show the research-results section, explain the original 84.57 percent test accuracy and weak trash recall, then return to slide 6.

If the app fails: use the PDF slides and saved reports in EVIDENCE. Explain the four-step workflow on slide 6 and say that the live demo is unavailable. Do not pretend it ran. The app uses saved weights and does not need retraining.

Evidence shortcuts: intro_sample_predictions.csv for Wednesday; regression_predictions.csv for Thursday; comparison.csv and mobilenet_confusion.png for Friday; WasteSort_Colab_Executed.ipynb for full cloud training. The Tuesday local executed notebook is included. Its separate cloud run was not yet saved and verified when this pack was prepared.

## 5. Likely questions and short answers

AI versus ML? AI is the broader field. Machine learning learns patterns from examples. Deep learning uses multilayer neural networks within ML.

Learning types? Supervised learning uses labeled examples, such as spam classification. Unsupervised learning finds structure without target labels, such as customer clusters. Reinforcement learning uses rewards to learn actions, such as game strategies.

Why supervised classification? Each image has a material label, and the answer is a category. Label codes 0-5 identify categories, not quantities.

Why separate data partitions? Training fits the model, validation chooses settings and the model, and testing evaluates the frozen choice. Repeatedly tuning to test scores undermines that evaluation.

KNN and Decision Tree results? Validation macro F1 was 57.03 percent for KNN and 35.04 percent for Decision Tree. Their predictions and confusion matrices are saved. Do not rank validation scores against other models' test scores.

Regression results? MSE was 2,900.19, the average squared prediction error. R-squared was 0.453, about 45.3 percent of test-target variation explained relative to a constant test-mean reference. It is not classification accuracy. This was a separate educational exercise.

Why MobileNetV2? It had the strongest validation macro F1 among the main models under our protocol. This does not prove it is always the best model.

What changed from the proposal? PyTorch replaced TensorFlow/Keras, candidate duplicates were grouped in an approximately 70/15/15 split, and we report descriptive comparisons rather than formal hypothesis rejection. These are documented for instructor review; approval is not recorded.

Did you use AI? Yes. ChatGPT/Codex assisted code, experiments, visual screening and documents. The group directed work through this shared chat. We disclose that assistance and must understand the methods.

Is the entire research finished? The app and initial benchmark are implemented and weekly outputs are available. Campus validation, instructor acceptance, full research chapters and independent human verification remain separate work. Sunday was an added verification day, not part of the original Monday-Saturday schedule.
