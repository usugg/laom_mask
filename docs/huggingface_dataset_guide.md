# HuggingFace Dataset Integration für LAPO

## Übersicht

Die LAPO-Implementierung unterstützt jetzt das Laden von Daten direkt von HuggingFace Hub, was mehrere Vorteile gegenüber HDF5-Dateien bietet:

- **Streaming-Unterstützung**: Kein vollständiger Download erforderlich
- **Automatisches Caching**: Daten werden lokal gecacht
- **Einfacher Zugriff**: Verschiedene Distraction-Levels per Konfiguration
- **Weniger Speicherplatz**: Kein doppelter Speicher für HDF5-Konvertierung

## Verfügbare Datensätze

Der Datensatz `EpicPinkPenguin/visual_distracting_control_suite` enthält Daten für verschiedene Umgebungen und Distraction-Levels:

### Umgebungen
- `cheetah_run`
- `walker_walk`
- `hopper_hop`
- `humanoid_walk`

### Distraction Levels
- `distractor_none`: Vanilla DMC ohne visuelle Ablenkungen
- `distractor_low`: Dynamischer Hintergrund, feste Kamera
- `distractor_hard`: Dynamischer Hintergrund + dynamische Kamera

### Konfigurationsnamen
Format: `{environment}_{distractor_level}`

Beispiele:
- `cheetah_run_distractor_hard`
- `walker_walk_distractor_low`
- `humanoid_walk

## Verwendung

### 1. Mit HuggingFace Dataset (empfohlen)

```bash
python train_lapo.py --config_path configs/lapo_hf_example.yaml
```

Oder mit Command-Line-Argumenten:

```bash
python train_lapo.py \
  --lapo.use_hf_dataset=true \
  --lapo.hf_config_name=cheetah_run_distractor_hard \
  --lapo.hf_streaming=true \
  --lapo.batch_size=256 \
  --lapo.num_epochs=100
```

### 2. Mit HDF5 (legacy)

```bash
python train_lapo.py \
  --lapo.use_hf_dataset=false \
  --lapo.data_path=data/your_dataset.hdf5
```

## Konfigurationsoptionen

### HuggingFace Dataset Parameter

```yaml
lapo:
  use_hf_dataset: true  # HuggingFace aktivieren
  hf_dataset_name: "EpicPinkPenguin/visual_distracting_control_suite"
  hf_config_name: "cheetah_run_distractor_hard"  # Umgebung + Distraction Level
  hf_split: "train"  # oder "test"
  hf_streaming: true  # Streaming aktivieren (empfohlen)
  hf_buffer_size: 10000  # Puffergröße für Frame-Stacking
```

### Wichtige Parameter

- **`use_hf_dataset`**: `true` für HuggingFace, `false` für HDF5
- **`hf_streaming`**: `true` = Streaming (weniger Speicher), `false` = vollständiger Download
- **`hf_buffer_size`**: Größe des Puffers für Frame-Stacking (Standard: 10000)
- **`frame_stack`**: Anzahl der zu stapelnden Frames (Standard: 3)
- **`future_obs_offset`**: Maximaler Offset für zukünftige Beobachtungen (Standard: 1)

## Dataset-Struktur

Jedes Sample enthält:
- **observation**: RGB-Bild (H, W, 3)
- **action**: Kontinuierliche Aktionen
- **state**: Zustandsvektor
- **mask**: Segmentierungsmaske des Agenten
- **reward**: Belohnung
- **terminated**: Episode beendet
- **truncated**: Episode abgebrochen

## Beispiel-Konfigurationen

### Cheetah Run - Hard Distraction
```yaml
lapo:
  use_hf_dataset: true
  hf_config_name: "cheetah_run_distractor_hard"
  batch_size: 256
  num_epochs: 100
```

### Walker Walk - Low Distraction
```yaml
lapo:
  use_hf_dataset: true
  hf_config_name: "walker_walk_distractor_low"
  batch_size: 256
  num_epochs: 100
```

### Humanoid Walk - No Distraction
```yaml
lapo:
  use_hf_dataset: true
  hf_config_name: "humanoid_walk"
  batch_size: 256
  num_epochs: 100
```

## Performance-Tipps

1. **Streaming aktivieren**: Spart Speicherplatz und Downloadzeit
2. **Buffer-Größe anpassen**: Größerer Buffer = mehr Speicher, aber bessere Performance
3. **Batch-Größe**: Bei Streaming kann die Batch-Größe die Geschwindigkeit beeinflussen
4. **GPU-Nutzung**: Daten werden direkt auf dem konfigurierten Device (GPU) geladen

## Troubleshooting

### Langsamer Download
- Streaming aktivieren: `hf_streaming: true`
- HuggingFace-Token verwenden für schnelleren Zugriff

### Out of Memory
- Buffer-Größe reduzieren: `hf_buffer_size: 5000`
- Batch-Größe reduzieren: `batch_size: 128`
- Frame-Stack reduzieren: `frame_stack: 1`

### Datensatz nicht gefunden
- Überprüfen Sie den Konfigurationsnamen
- Verfügbare Konfigurationen: https://huggingface.co/datasets/EpicPinkPenguin/visual_distracting_control_suite

## Implementation Details

### DCSLAPOHFDataset
- Erbt von `IterableDataset` für Streaming-Unterstützung
- Implementiert Frame-Stacking mit Puffer
- Unterstützt zufällige Offsets für zukünftige Beobachtungen
- Lädt Daten direkt auf GPU/CPU basierend auf `device` Parameter

### DCSLAPOInMemoryDataset
- Erbt von `Dataset` für HDF5-Dateien
- Lädt alle Daten in den Speicher
- Unterstützt Shuffling und Drop-Last
