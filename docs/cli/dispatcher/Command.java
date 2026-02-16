package docs.cli.dispatcher;

import java.util.List;

/**
 * Shared contract for all commands that can be registered with the Dispatcher.
 */
public interface Command {
    void execute(List<String> args) throws CommandException;
}
