import chartDiscountedReward as cdr
import sys
import glob


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python chartDiscountedReward.py <file_path_glob> <save_file>")
        sys.exit(1)

    matching_file_paths = glob.glob(sys.argv[1])
    save_path = sys.argv[2]

    print("Matching Files: ", matching_file_paths)
    print("Save Path: ", save_path)
    cdr.merge_and_save(matching_file_paths, save_path)
