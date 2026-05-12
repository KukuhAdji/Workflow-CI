# Workflow CI - MLflow Project

Folder ini untuk kriteria 3.

## Local test
```bash
cd Workflow-CI
mlflow run MLProject -P data_dir=namadataset_preprocessing -P target_col=target
```

## GitHub
1. Buat repository baru bernama `Workflow-CI`.
2. Upload isi folder ini.
3. Pastikan dataset preprocessing ada di `MLProject/namadataset_preprocessing`.
4. Workflow berada di `.github/workflows/train.yml`.
5. Jalankan GitHub Actions dari tab Actions.
6. Pastikan folder `training-artifacts` muncul/ter-commit atau artifact workflow tersedia.

## Untuk Advanced Docker Hub
Tambahkan secret:
- DOCKERHUB_USERNAME
- DOCKERHUB_TOKEN

Lalu aktifkan langkah Docker pada workflow sesuai komentar.
