.PHONY: help init build clean run test test-patterns benchmark image lint fmt

DEFAULT_GOAL := help
APP_NAME := cu

# init
init:
	@prek install --hook-type commit-msg --hook-type pre-commit --hook-type pre-push

# configure
configure:
	@cmake -S . -B build

# build
build: configure
	@cmake --build build --parallel

# clean
clean:
	@rm -rf build

# run demo
run: build
	@./build/bin/$(APP_NAME)

# run all tests
test: build
	@./build/bin/cu-tests
	@./build/bin/test_patterns-tests

# run unit tests only
test-unit: build
	@./build/bin/cu-tests

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
	@echo "Usage:"
	@echo "    make [target]"
	@echo ""
	@echo "Targets:"
	@awk '/^[a-zA-Z\-_0-9]+:/ \
	{ \
		helpMessage = match(lastLine, /^# (.*)/); \
		if (helpMessage) { \
			helpCommand = substr($$1, 0, index($$1, ":")-1); \
			helpMessage = substr(lastLine, RSTART + 2, RLENGTH); \
			printf "\033[36m%-22s\033[0m %s\n", helpCommand,helpMessage; \
		} \
	} { lastLine = $$0 }' $(MAKEFILE_LIST)
