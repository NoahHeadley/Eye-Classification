import argparse


def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--clusters", required=True,
                    help="path to clustered data")
    ap.add_argument("-i", "--image", required=True,
                    help="image file to analyze")
    return vars(ap.parse_args())


if __name__ == '__main__':
    args = build_arg_parser()
    cluster_file = args['clusters']
    image = args['image']

    cluster_values = open(cluster_file, 'r')
    lines = cluster_values.readlines()
    values = []
    for line in lines:
        try:
            line = [float(i) for i in line.split()]
            values.append(line)
        except:
            next

    num_vals, num_vars, _ = values[0]
    num_vals, num_vars = int(num_vals), int(num_vars)
    clusters = []
    for i in range(1, num_vals+1):
        if values[i][num_vars] not in clusters:
            clusters.append(int(values[i][num_vars]))
    num_clusts = int(max(clusters))

    cluster_mega = []
    for i in range(1, num_vals+1):
        cluster = int(values[i][num_vars]) - 1
        cluster_mega.append((values[i][:num_vars], cluster))

    for val in cluster_mega:
        x, y = val
        if(y == 2):
            print(x)
