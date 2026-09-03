/**
 * @file graph_algorithms.cpp
 * @brief Graph algorithm example: BFS + PageRank over a generated random graph
 *        — ported to the current cuda::graph API (TASK-064; the old
 *        nova::graph::* API / CSRGraph(raw CSR arrays) ctor no longer exists;
 *        graphs are now built from an edge list via create_csr_from_edges).
 * @example
 *
 * Compile (part of the example targets):
 *   cmake --build build --target graph_algorithms
 *
 * Run:
 *   ./build/bin/graph_algorithms --algorithm bfs --nodes 10000 --edges 50000
 *   ./build/bin/graph_algorithms --algorithm pagerank --nodes 10000 --iterations 20
 */

#include <cuda/graph/csr_graph.h>
#include <cuda/graph/bfs.h>
#include <cuda/graph/pagerank.h>
#include <cuda/device/error.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

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
    printf("  -h, --help          Show this help and exit\n");
    printf("  --algorithm <name>  Algorithm: bfs, pagerank\n");
    printf("  --nodes <n>         Number of nodes (positive int)\n");
    printf("  --edges <n>         Number of edges (positive int)\n");
    printf("  --source <n>        Source node for BFS\n");
    printf("  --iterations <n>    PageRank iterations window (positive int)\n");
}

// Positive-int parse with fail-fast validation (the old atoi silently
// accepted garbage as 0).
bool parse_positive_int(const char* arg, int* out, const char* name) {
    char* end = nullptr;
    const long v = strtol(arg, &end, 10);
    if (end == arg || *end != '\0' || v < 0 || v > 1e9) {
        fprintf(stderr, "Error: --%s expects a non-negative integer, got '%s'\n",
                name, arg);
        return false;
    }
    *out = static_cast<int>(v);
    return true;
}

// Returns 1 on --help (exit 0), 0 on success, -1 on error (exit 1).
int parse_args(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 1;
        }
        if (strcmp(argv[i], "--algorithm") == 0 && i + 1 < argc) {
            args.algorithm = argv[++i];
        } else if (strcmp(argv[i], "--nodes") == 0 && i + 1 < argc) {
            if (!parse_positive_int(argv[++i], &args.nodes, "nodes")) return -1;
            if (args.nodes == 0) {
                fprintf(stderr, "Error: --nodes must be at least 1 (an empty "
                                "graph would index out of range)\n");
                return -1;
            }
        } else if (strcmp(argv[i], "--edges") == 0 && i + 1 < argc) {
            if (!parse_positive_int(argv[++i], &args.edges, "edges")) return -1;
        } else if (strcmp(argv[i], "--source") == 0 && i + 1 < argc) {
            if (!parse_positive_int(argv[++i], &args.source, "source")) return -1;
        } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            if (!parse_positive_int(argv[++i], &args.iterations, "iterations"))
                return -1;
        } else {
            fprintf(stderr, "Error: unknown or incomplete option '%s'\n", argv[i]);
            print_usage(argv[0]);
            return -1;
        }
    }
    if (strcmp(args.algorithm, "bfs") != 0 &&
        strcmp(args.algorithm, "pagerank") != 0) {
        fprintf(stderr, "Error: unknown algorithm '%s' (choose bfs | pagerank)\n",
                args.algorithm);
        return -1;
    }
    if (args.nodes > 0 && args.source >= args.nodes) {
        fprintf(stderr, "Error: --source %d out of range for %d nodes\n",
                args.source, args.nodes);
        return -1;
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    const int parse = parse_args(argc, argv, args);
    if (parse < 0) return 1;
    if (parse > 0) return 0;

    printf("Nova Graph Algorithms Example\n");
    printf("Algorithm: %s, Nodes: %d, Edges: %d\n",
           args.algorithm, args.nodes, args.edges);

    // Generate a random edge list; each node gets a light random degree.
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> node_dist(0, args.nodes - 1);
    std::vector<int> src(args.edges), dst(args.edges);
    std::vector<float> weights(args.edges, 1.0f);
    for (int i = 0; i < args.edges; ++i) {
        src[static_cast<size_t>(i)] = node_dist(rng);
        // Avoid self-loops for a cleaner demo graph.
        int d = node_dist(rng);
        dst[static_cast<size_t>(i)] = (d == src[static_cast<size_t>(i)])
                                          ? (d + 1) % args.nodes
                                          : d;
    }

    try {
        std::unique_ptr<cuda::graph::CSRGraph> graph =
            cuda::graph::create_csr_from_edges(src.data(), dst.data(),
                                              weights.data(), args.nodes,
                                              args.edges);

        if (strcmp(args.algorithm, "bfs") == 0) {
            printf("Running BFS from source node %d...\n", args.source);
            cuda::graph::BFSResult result = cuda::graph::bfs(*graph, args.source);
            int visited = 0;
            for (int v = 0; v < args.nodes; ++v) {
                if (result.is_reachable(v)) ++visited;
            }
            printf("BFS complete: Visited %d/%d nodes, max distance %d\n",
                   visited, args.nodes, result.max_distance);
            printf("First 10 distances: ");
            for (int v = 0; v < std::min(10, args.nodes); ++v) {
                printf("%d ", result.distance_to(v));
            }
            printf("\n");
        } else {
            printf("Running PageRank (up to %d iterations, damping %.2f)...\n",
                   args.iterations, args.damping);
            cuda::graph::PageRankOptions options;
            options.damping = args.damping;
            options.tolerance = args.tolerance;
            options.max_iterations = args.iterations;
            cuda::graph::PageRankResult result =
                cuda::graph::pagerank(*graph, options);
            const int k = std::min(5, args.nodes);
            std::vector<int> top = result.top_k(k);
            printf("PageRank converged in %d iterations (final delta %g)\n",
                   result.iterations, result.final_delta);
            printf("Top %d pages: ", k);
            for (int i = 0; i < k; ++i) {
                printf("%d(%.3f) ", top[static_cast<size_t>(i)],
                       result.rank_at(top[static_cast<size_t>(i)]));
            }
            printf("\n");
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
    return 0;
}
