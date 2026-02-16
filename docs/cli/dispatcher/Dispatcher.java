package docs.cli.dispatcher;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Hashtable;
import java.util.List;

/**
 * Main dispatcher that routes first-token command names to command objects.
 */
public class Dispatcher {
    private final Hashtable<String, Command> commandRegistry = new Hashtable<>();

    public Dispatcher() {
        // Register commands here. Extensibility is achieved by adding more entries.
        register("greet", new GreetCommand());
    }

    public void register(String token, Command command) {
        commandRegistry.put(token, command);
    }

    public void dispatch(String[] rawArgs) {
        ParsedInput parsedInput = parseGlobalOptions(rawArgs);

        if (parsedInput.showHelp || parsedInput.commandToken == null) {
            usage();
            return;
        }

        Command command = commandRegistry.get(parsedInput.commandToken);
        if (command == null) {
            System.err.printf("Unknown command: %s%n", parsedInput.commandToken);
            usage();
            return;
        }

        try {
            command.execute(parsedInput.commandArgs);
        } catch (CommandException ex) {
            System.err.printf("Command '%s' failed: %s%n", parsedInput.commandToken, ex.getMessage());
            usage();
        } catch (RuntimeException ex) {
            // Catch unexpected runtime errors and surface a friendly message.
            System.err.printf("Unexpected error while running '%s': %s%n", parsedInput.commandToken, ex.getMessage());
        }
    }

    /**
     * Simplified "getopts"-style global option parsing.
     *
     * Supported global options:
     *   -h, --help      Show usage.
     *   --verbose       Example global flag (available for future expansion).
     */
    private ParsedInput parseGlobalOptions(String[] rawArgs) {
        ParsedInput parsed = new ParsedInput();
        List<String> tokens = new ArrayList<>(Arrays.asList(rawArgs));

        while (!tokens.isEmpty() && tokens.get(0).startsWith("-")) {
            String option = tokens.remove(0);
            switch (option) {
                case "-h":
                case "--help":
                    parsed.showHelp = true;
                    return parsed;
                case "--verbose":
                    parsed.verbose = true;
                    break;
                default:
                    System.err.printf("Unknown global option: %s%n", option);
                    parsed.showHelp = true;
                    return parsed;
            }
        }

        if (!tokens.isEmpty()) {
            parsed.commandToken = tokens.remove(0);
            parsed.commandArgs = tokens;
        }

        return parsed;
    }

    private void usage() {
        System.err.println("Usage: <program> [global-options] <command> [command-args]");
        System.err.println("Global options:");
        System.err.println("  -h, --help      Show this help text");
        System.err.println("  --verbose       Enable verbose output");
        System.err.println("Commands:");
        System.err.println("  greet <name>    Print a greeting for <name>");
    }

    public static void main(String[] args) {
        Dispatcher dispatcher = new Dispatcher();
        dispatcher.dispatch(args);
    }

    private static class ParsedInput {
        private boolean showHelp;
        private boolean verbose;
        private String commandToken;
        private List<String> commandArgs = new ArrayList<>();
    }
}
