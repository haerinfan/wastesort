# Running WasteSort

## Windows: start here

1. Install Python **3.12** from [python.org](https://www.python.org/downloads/). Include the Python launcher when installing.
2. On the [repository page](https://github.com/haerinfan/wastesort), choose **Code → Download ZIP** and extract the entire ZIP.
3. Open the extracted folder containing `app.py` and double-click **Start WasteSort.cmd**.
4. First setup downloads the pinned dependencies. Later starts reuse them. Keep the window open.
5. Open **http://localhost:8501**, upload one item photo, and select **Identify material**.

The model is bundled. No training, account, API key, GPU or dataset download is needed to classify a photo. Initial dependency installation needs internet; local inference works offline afterward. The launcher binds the app to this computer only and disables Streamlit usage statistics. Press Ctrl+C in the terminal to stop.

## Terminal alternative

From the extracted project folder:

```powershell
py -3.12 scripts/launch.py
```

Check setup without starting the browser:

```powershell
py -3.12 scripts/launch.py --check
```

On macOS/Linux use `python3.12 scripts/launch.py` instead. This path is implemented but has not been manually verified on those operating systems.

## Common problems

| Message or symptom | Next step |
|---|---|
| `py` is not recognized or Python 3.12 is missing | Install Python 3.12 with its launcher, then reopen the terminal. |
| Dependency download stops | Check internet access, then rerun the launcher. It only records successful installation. |
| Installed packages were changed | Run `py -3.12 scripts/launch.py --reinstall`. |
| Model checksum or missing model error | Extract a fresh complete ZIP; keep the `models` folder next to `app.py`. |
| Port 8501 is already in use | Stop the earlier app with Ctrl+C, or use `py -3.12 scripts/launch.py --port 8502` and open the printed address. |
| Browser did not open | Enter the printed localhost address in your browser. |
| Photo is rejected | Use an actual JPEG/PNG under 10 MB, at least 32 pixels on each side and at most 20 million pixels. |

For an existing environment, `python -m wastesort.doctor` checks dependency versions, model integrity, label order and a real CPU inference. It does not change the dataset or model.

The app gives material predictions, not verified campus disposal instructions. Its model score is uncalibrated; unrelated objects can still receive a waste label.

## Save predictions without a browser

After setup, run this in the project folder, replacing the example photo paths with your own:

```powershell
.\.venv\Scripts\python.exe predict.py "C:\Photos\bottle.jpg" "C:\Photos\box.png" --output predictions.json
```

Use an output name ending in `.csv` for a spreadsheet report. Existing reports are never overwritten. Invalid photos are recorded as errors while valid photos still receive predictions; a run with any errors returns a nonzero exit code. JSON includes the model checksum and timestamp. CSV protects text cells from spreadsheet formula interpretation. Reports include supplied file paths, so review them before sharing. Photos themselves are not copied into reports.

These are prediction records, not an accuracy evaluation. Use separately collected, permissioned and labeled photos for any future campus evaluation, and report it separately from the published TrashNet study.
