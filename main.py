from ml_config import MLModelLoader

def main():
    loader = MLModelLoader("ml_config.yml")
    asl = loader.get("sign_detect")
    asl.run_realtime()

if __name__ == "__main__":
    main()
