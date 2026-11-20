.PHONY: help init build build-ninja configure configure-ninja clean clean-build clean-ninja \
	run run-ninja test test-ninja test-one test-patterns benchmark image \
	lint fmt fmt-check sanitize prek

DEFAULT_GOAL := help
APP_NAME := nova
NINJA_BUILD := build-ninja

# Every source file clang-format and clang-tidy cover. Shared by the fmt and
# fmt-check targets so the two can never drift apart.
SOURCE_DIRS := src include tests benchmark
SOURCE_GLOBS := -name '*.cpp' -o -name '*.h' -o -name '*.hpp' -o -name '*.cu' -o -name '*.cuh'

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
	@echo "Quality Targets:"
	@echo "    make init               Install the prek git hooks"
	@echo "    make prek               Run all prek hooks over the whole tree"
	@echo "    make lint               Run clang-tidy static analysis"
	@echo "    make fmt                Apply clang-format in place"
	@echo "    make fmt-check          Check formatting with clang-format"
	@echo "    make sanitize           Build & test with ASan+UBSan"
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

# Run clang-tidy static analysis
#
# The config path is absolute on purpose: CMAKE_CXX_CLANG_TIDY is a command line,
# and clang-tidy resolves --config-file relative to the COMPILER WORKING DIRECTORY,
# which for a Ninja build is the build directory. A relative ".clang-tidy" made
# every translation unit fail with
#
#     Error: can't read config-file '.clang-tidy': No such file or directory
#
# so the target never reached the project's checks. $(CURDIR) is the Makefile's
# directory, which is where .clang-tidy actually lives.
lint:
	@cmake -G Ninja -B build-tidy \
		-DCMAKE_CXX_CLANG_TIDY="clang-tidy;--config-file=$(CURDIR)/.clang-tidy" \
		-DNOVA_ENABLE_NCCL=OFF -DNOVA_ENABLE_MPI=OFF
	@cmake --build build-tidy --parallel

# Check code formatting
fmt-check:
	@find $(SOURCE_DIRS) \( $(SOURCE_GLOBS) \) -type f -print0 \
		| xargs -0 clang-format --dry-run --Werror --style=file && echo "Format: OK"

# Apply clang-format in place. `fmt` was listed in .PHONY with no rule, so
# `make fmt` exited 0 having done nothing.
fmt:
	@find $(SOURCE_DIRS) \( $(SOURCE_GLOBS) \) -type f -print0 \
		| xargs -0 clang-format -i --style=file && echo "Format: applied"

# Run every pre-commit hook over the whole tree, without needing a commit.
prek:
	@prek run --all-files

# Build and test with AddressSanitizer + UndefinedBehaviorSanitizer
sanitize:
	@cmake -G Ninja -B build-san \
		-DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer" \
		-DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined" \
		-DNOVA_ENABLE_NCCL=OFF -DNOVA_ENABLE_MPI=OFF
	@cmake --build build-san --parallel
	@ASAN_OPTIONS=detect_leaks=1:detect_stack_use_after_return=1 \
		UBSAN_OPTIONS=print_stacktrace=1 \
		ctest --test-dir build-san --output-on-failure -j8
