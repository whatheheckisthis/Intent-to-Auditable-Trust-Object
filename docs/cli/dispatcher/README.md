# Modular CLI Dispatcher Example

This example demonstrates a command-driven dispatcher with:

- `Command` interface: `execute(List<String> args) throws CommandException`
- Registry based on `Hashtable<String, Command>`
- Simplified getopts-style global options parsing in the dispatcher
- `CommandException`-based error handling with centralized `usage()`
- Easy extensibility by registering new commands via `register(token, command)`

## Files

- `Command.java` - command interface
- `CommandException.java` - custom checked exception type
- `GreetCommand.java` - sample command implementation
- `Dispatcher.java` - main registry + routing + global option parser

## Compile and run

```bash
javac docs/cli/dispatcher/*.java
java docs.cli.dispatcher.Dispatcher greet Alice
```

## IAM policy parser (Python)

Production parser entrypoint: `docs/cli/dispatcher/iam_policy_parser.py` (stdout emits one JSON object only).

