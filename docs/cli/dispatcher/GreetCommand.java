package docs.cli.dispatcher;

import java.util.List;

/**
 * Example command implementation.
 *
 * Usage:
 *   greet <name>
 */
public class GreetCommand implements Command {
    @Override
    public void execute(List<String> args) throws CommandException {
        if (args.isEmpty()) {
            throw new CommandException("Missing required argument: <name>");
        }

        String name = args.get(0);
        System.out.printf("Hello, %s!%n", name);
    }
}
