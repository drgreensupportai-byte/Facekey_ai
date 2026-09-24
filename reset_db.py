import os
import shutil
from app import app, db, User  # Hindura niba app yawe ifite imiterere itandukanye
from config import Config

def reset_system():
    print("⚠️ Kutangira gusiba amakuru yose...")

    with app.app_context():
        # 1. Siba amakuru yose ari mu shingiro rya Data (Database Tables)
        try:
            num_rows = db.session.query(User).delete()
            db.session.commit()
            print(f"✓ Abakoresha (Users) {num_rows} basibiwe mu database.")
        except Exception as e:
            db.session.rollback()
            print(f"❌ Ikosa mu gusiba database: {e}")

    # 2. Siba amafoto ari muri folder ya DATASET
    dataset_dir = getattr(Config, 'DATASET_DIR', 'dataset')
    if os.path.exists(dataset_dir):
        for filename in os.listdir(dataset_dir):
            file_path = os.path.join(dataset_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"❌ Ntiwabashije gusiba {file_path}: {e}")
        print("✓ Amafoto yose yo muri dataset yasibiwe.")

    # 3. Siba FAISS Vector Index File
    faiss_path = getattr(Config, 'FAISS_INDEX_PATH', os.path.join('indexes', 'facekey_faiss.index'))
    if os.path.exists(faiss_path):
        try:
            os.remove(faiss_path)
            print("✓ FAISS Vector Index yasibiwe.")
        except Exception as e:
            print(f"❌ Ikosa mu gusiba FAISS index: {e}")

    print("\n🎉 Amakuru yose yasibiwe neza! Sisitemu irasubiye mu miterere y'intangiriro.")

if __name__ == '__main__':
    confirm = input("Uramutse utabyizeye kanda 'Y' kugira ngo usibe amakuru yose (Y/N): ")
    if confirm.lower() == 'y':
        reset_system()
    else:
        print("Ibikorwa byahagaritswe.")