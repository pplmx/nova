/**
 * @file graph_algorithms.cpp
 * @brief Graph algorithm example demonstrating BFS and PageRank
 * @example
 *
 * Compile:
 *   g++ -std=c++23 -I include examples/graph_algorithms.cpp \
 *       -L build/lib -lcuda_impl -lcudart -o graph_algorithms
 *
 * Run:
 *   ./graph_algorithms --algorithm bfs --nodes 10000 --edges 50000
 */

#include <cuda/graph/csr_graph.hpp>
#include <cuda/graph/bfs.hpp>
#include <cuda/graph/pagerank.hpp>
#include <cuda/error/cuda_error.hpp>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <random>

struct Args {
    const char* algorithm = "bfs";
    int nodes = 10000;
    int edges = 50000;
    int source = 0;
    int iterations = 20;
    float damping = 0.85f;
    float tolerance = 1e-6f;
};

void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  --algorithm <name>   Algorithm: bfs, pagerank\n");
    printf("  --nodes <n>          Number of nodes\n");
    printf("  --edges <n>          Number of edges\n");
    printf("  --source <n>         Source node for BFS\n");
    printf("  --iterations <n>     PageRank iterations\n");
}

int main(int argc, char** argv) {
    Args args;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--algorithm") == 0 && i + 1 < argc) {
            args.algorithm = argv[++i];
        } else if (strcmp(argv[i], "--nodes") == 0 && i + 1 < argc) {
            args.nodes = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--edges") == 0 && i + 1 < argc) {
            args.edges = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--source") == 0 && i + 1 < argc) {
            args.source = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            args.iterations = atoi(argv[++i]);
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    printf("Nova Graph Algorithms Example\n");
    printf("Algorithm: %s, Nodes: %d, Edges: %d\n",
           args.algorithm, args.nodes, args.edges);

    // Generate random graph
    std::vector<int> row_offsets(args.nodes + 1);
    std::vector<int> columns(args.edges);
    std::vector<float> weights(args.edges);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, args.nodes - 1);

    // Generate edges
    row_offsets[0] = 0;
    for (int i = 0; i < args.nodes; i++) {
        int degree = std::max(1, args.edges / args.nodes);
        row_offsets[i + 1] = row_offsets[i] + degree;
        for (int j = row_offsets[i]; j < row_offsets[i + 1]; j++) {
            columns[j] = dist(rng);
            weights[j] = 1.0f / degree;
        }
    }

    // Create CUDA graph
    nova::graph::CSRGraph graph(args.nodes, args.edges,
                               row_offsets.data(),
                               columns.data(),
                               weights.data());

    if (strcmp(args.algorithm, "bfs") == 0) {
        printf("Running BFS from source node %d...\n", args.source);

        std::vector<int> distances(args.nodes, -1);
        nova::memory::Buffer<int> d_distances(args.nodes);
        d_distances.copy_from_host(distances.data());

        nova::graph::bfs(graph, args.source, d_distances.device_data());

        d_distances.copy_to_host(distances.data());

        // Count visited nodes
        int visited = 0;
        for (int i = 0; i < args.nodes; i++) {
            if (distances[i] >= 0) visited++;
        }
        printf("BFS complete: Visited %d/%d nodes\n", visited, args.nodes);

        // Print first 10 distances
        printf("First 10 distances: ");
        for (int i = 0; i < std::min(10, args.nodes); i++) {
            printf("%d ", distances[i]);
        }
        printf("\n");

    } else if (strcmp(args.algorithm, "pagerank") == 0) {
        printf("Running PageRank with %d iterations...\n", args.iterations);

        std::vector<float> ranks(args.nodes, 1.0f / args.nodes);
        nova::memory::Buffer<float> d_ranks(args.nodes);
        d_ranks.copy_from_host(ranks.data());

        for (int i = 0; i < args.iterations; i++) {
            nova::graph::pagerank(
                graph,
                d_ranks.device_data(),
                args.damping,
                args.tolerance
            );
        }

        d_ranks.copy_to_host(ranks.data());

        // Find top 10 pages
        std::vector<std::pair<int, float>> page_scores(args.nodes);
        for (int i = 0; i < args.nodes; i++) {
            page_scores[i] = {i, ranks[i]};
        }
        std::sort(page_scores.begin(), page_scores.end(),
                 [](const auto& a, const auto& b) { return a.second > b.second; });

        printf("Top 10 PageRank scores:\n");
        for (int i = 0; i < 10; i++) {
            printf("  Node %d: %.6f\n", page_scores[i].first, page_scores[i].second);
        }
    }

    printf("Graph algorithm complete!\n");
    return 0;
}
