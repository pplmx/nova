.PHONY: help init build build-ninja configure clean run test test-patterns benchmark image lint fmt

DEFAULT_GOAL := help
APP_NAME := nova
NINJA_BUILD := build-ninja

# init
init:
	@prek install --hook-type commit-msg --hook-type pre-commit --hook-type pre-push

# configure (Make generator)
configure:
	@cmake -S . -B build

# configure (Ninja generator - faster)
configure-ninja:
	@cmake -G Ninja -S . -B $(NINJA_BUILD)

# build (Make generator - uses all CPU cores)
build: configure
	@cmake --build build --parallel

# build (Ninja generator - recommended for speed)
build-ninja: configure-ninja
	@cmake --build $(NINJA_BUILD) --parallel

# clean all builds
clean:
	@rm -rf build $(NINJA_BUILD)

# clean specific build
clean-build:
	@rm -rf build

clean-ninja:
	@rm -rf $(NINJA_BUILD)

# run demo
run: build
	@./build/bin/$(APP_NAME)

# run demo (Ninja build)
run-ninja: build-ninja
	@./$(NINJA_BUILD)/bin/$(APP_NAME)

# run all tests (parallel - uses NCPU cores, capped at 16 for GPU memory)
test: build
	@cd build && ctest -j16 --output-on-failure

# run all tests (Ninja build)
test-ninja: build-ninja
	@cd $(NINJA_BUILD) && ctest -j16 --output-on-failure

# run single test
test-one: build
	@./build/bin/nova-tests --gtest_filter="$(filter)"
	@echo "Usage: make test-one filter='TestSuite.TestName'"

# run pattern tests only
test-patterns: build
	@./build/bin/test_patterns-tests

# run benchmark demo
benchmark: build
	@./build/bin/$(APP_NAME)

# build docker image
image:
	@docker image build -t $(APP_NAME) .

# Show help
help:
	@echo ""
	@echo "Nova CUDA Library - Build System"
	@echo ""
	@echo "Usage:"
	@echo "    make [target]"
	@echo ""
	@echo "Build Targets:"
	@echo "    make build              Build with Make (all CPU cores)"
	@echo "    make build-ninja        Build with Ninja (fastest, recommended)"
	@echo "    make configure          Configure for Make"
	@echo "    make configure-ninja    Configure for Ninja"
	@echo "    make clean              Clean all builds"
	@echo "    make clean-ninja        Clean Ninja build only"
	@echo ""
	@echo "Run Targets:"
	@echo "    make run                Run demo (Make build)"
	@echo "    make run-ninja          Run demo (Ninja build)"
	@echo "    make test               Run tests (16 parallel)"
	@echo "    make test-ninja         Run tests (Ninja build)"
	@echo ""
	@echo "Options:"
	@echo "    NOVA_ENABLE_NCCL=OFF     Disable NCCL (no GPU required)"
	@echo "    NOVA_ENABLE_MPI=OFF      Disable MPI (single node only)"
	@echo ""
	@echo "Example (fastest build):"
	@echo "    make configure-ninja NOVA_ENABLE_NCCL=OFF"
	@echo "    make build-ninja"
	@echo "    make test-ninja"
	@echo ""
