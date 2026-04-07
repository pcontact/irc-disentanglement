# Transcript Graph Visualizer

This is a small browser-side visualizer for transcript JSONL files shaped like:

```json
{"id": 0, "message": "hello", "outgoing_edge_target_id": -1}
```

## Run locally

From the repository root:

```bash
python3 -m http.server 8000
```

Then open:

```text
http://localhost:8000/visualizer/
```

## Use it

1. Drag a transcript `.jsonl` file into the page, or click the drop zone to choose one.
2. Hover a row to highlight its local reply neighborhood.
3. Click a row to pin its full connected conversation.
4. Use `Clear Selection` to remove the pin.

## Notes

- The app is dependency-free: plain HTML, CSS, and JavaScript.
- Files are parsed entirely in the browser with no backend upload step.
- Invalid targets, duplicate ids, and parse errors are surfaced in the validation panel.
