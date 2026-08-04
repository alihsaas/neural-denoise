from neural_denoise.datasets import ensure_sidd, write_download_record


def main() -> None:
    record = ensure_sidd()
    write_download_record(record)
    for name, details in record.items():
        print(f"{name}: {details['bytes']} bytes, sha256 {details['sha256']}")


if __name__ == "__main__":
    main()
