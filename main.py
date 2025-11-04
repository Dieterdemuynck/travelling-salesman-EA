import r0123456 as tsp
import argparse

def main():
    parser = argparse.ArgumentParser(description='Solve Traveling Salesman Problem using Evolutionary Algorithm')
    parser.add_argument('file_path', help='Path to the TSP data file')
    args = parser.parse_args()
    
    tsp.r0123456().optimize(args.file_path)

if __name__ == "__main__":
    main()
