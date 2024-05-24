import argparse
import train_utils
import os

def main():
    results_dict = {}
    for graph_size in range(5,10):
        output_dir = train_utils.resolve_output_dirs(config, graph_size)
        
        try:
            latest_result_file = sorted(os.listdir(output_dir))[-2] if graph_size > 6 and len(os.listdir(output_dir)) > 1 else sorted(os.listdir(output_dir))[-1] 
        except:
            break
         
        results = train_utils.load_json(os.path.join(output_dir, latest_result_file))[0]
        
        for metric in results:
            if not "confusion_matrix" in metric:
                if metric not in results_dict:
                    results_dict[metric] = []
                
                results_dict[metric].append(results[metric])
    
    print(results_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read configuration JSON file')
    parser.add_argument('config', type=str, help='Path to the configuration JSON file')
    args = parser.parse_args()
    
    global config
    config = train_utils.load_json(args.config)
    config.update(args.__dict__)
    config = argparse.Namespace(**config)
    main()