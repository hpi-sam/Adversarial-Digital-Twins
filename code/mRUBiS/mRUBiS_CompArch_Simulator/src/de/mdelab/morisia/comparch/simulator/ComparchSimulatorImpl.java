package de.mdelab.morisia.comparch.simulator;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.logging.ConsoleHandler;
import java.util.logging.FileHandler;
import java.util.logging.Formatter;
import java.util.logging.Level;
import java.util.logging.LogRecord;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;

import de.mdelab.morisia.comparch.ArchitecturalElement;
import de.mdelab.morisia.comparch.Architecture;
import de.mdelab.morisia.comparch.simulator.impl.InjectionController;
import de.mdelab.morisia.comparch.simulator.impl.SimulatorUtil;
import de.mdelab.morisia.comparch.simulator.mrubis.events.ChangeEventObserver;
import de.mdelab.morisia.comparch.simulator.mrubis.events.ChangeEventQueue;

/**
 * Implementation of {@link ComparchSimulator}.
 * 
 * @author thomas vogel
 * @version 0.30
 *
 */
public class ComparchSimulatorImpl implements ComparchSimulator {

	private final static Logger LOGGER = Logger
			.getLogger(ComparchSimulatorImpl.class.getName());

	// params
	private Architecture eArchitecture;
	private int runs;
	private List<Injector<? extends ArchitecturalElement>> injectors;
	private List<ModelAnalyzer> modelAnalyzers;
	private List<AdaptationAnalyzer> adaptationAnalyzers;

	// info maintained
	private int runCount;


	// simulator classes
	private InjectionController injectionController;

	/**
	 * See
	 * {@link ComparchSimulatorFactory#createSimulator(Capability, Architecture, int, Level, String, boolean)}
	 * .
	 * 
	 * @param eArchitecture
	 * @param runs
	 * @param logLevel
	 * @param logFile
	 * @param logToConsole
	 */
	ComparchSimulatorImpl(Architecture eArchitecture, int runs,Level logLevel,
			String logFile, boolean logToConsole) {

		this.configLogging(logLevel, logFile, logToConsole);

		this.eArchitecture = eArchitecture;
		// observe change events
		this.eArchitecture.eAdapters().add(new ChangeEventObserver());

		this.runs = runs;
		this.injectors = new LinkedList<Injector<? extends ArchitecturalElement>>();
		this.modelAnalyzers = new LinkedList<ModelAnalyzer>();
		this.adaptationAnalyzers = new LinkedList<AdaptationAnalyzer>();

		this.runCount = 0;
		

		this.injectionController = new InjectionController(this.injectors);

	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public void setInjectionStrategy(InjectionStrategy strategy) {
		if (this.runCount == 0) {
			this.injectionController.setInjectionStrategy(strategy);
		} else {
			LOGGER.warning("Simulation is alredy running. The injection strategy cannot be changed any more.");
		}
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public List<Injector<? extends ArchitecturalElement>> getInjectors() {
		return Collections.unmodifiableList(this.injectors);
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public IssueType[] getSupportedIssueTypes() {
		int numberOfIssueTypes = this.injectors.size();
		ArrayList<IssueType> issueTypeList = new ArrayList<IssueType>(
				numberOfIssueTypes);
		this.injectors
				.forEach((Injector<? extends ArchitecturalElement> i) -> issueTypeList
						.add(i.getIssueType()));
		return issueTypeList.toArray(new IssueType[numberOfIssueTypes]);
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public boolean isSimulationCompleted() {
		return this.runCount == this.runs;
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public void addInjector(Injector<? extends ArchitecturalElement> injector) {
		if (this.runCount == 0) {
			this.injectors.add(injector);
		} else {
			LOGGER.warning("Simulation is alredy running. Injectors cannot be added any more.");
		}
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public void addAnalyzer(Analyzer analyzer) {
		if (this.runCount == 0) {
			if (analyzer instanceof ModelAnalyzer) {
				this.modelAnalyzers.add((ModelAnalyzer) analyzer);
			} else if (analyzer instanceof AdaptationAnalyzer) {
				this.adaptationAnalyzers.add((AdaptationAnalyzer) analyzer);
			}
		} else {
			LOGGER.warning("Simulation is alredy running. Analyzers cannot be added any more.");
		}
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public int validate() {
		if (this.modelAnalyzers.size() == 0) {
			throw new RuntimeException(
					"No model analyzers added to the simulator.");
		} // adaptation analyzers might not be required to run the simulator.

		LOGGER.info("\n\n\n**********************************************************************************************\n"
				+ "**********************************************************************************************\n"
				+ "\tAnalyze the adaptation and model ..."
				+ "\n**********************************************************************************************\n"
				+ "**********************************************************************************************");

		int failureCount = 0;

		// analyze the adaptation
		for (Analyzer analyzer : this.adaptationAnalyzers) {
			LOGGER.info(SimulatorUtil.SEPARATOR + "\t" + analyzer.getName()
					+ ": " + analyzer.getDescription()
					+ SimulatorUtil.SEPARATOR);
			int failures = analyzer.analyze(this.eArchitecture);
			failureCount = failureCount + failures;
		}

		// analyze the model
		for (Analyzer analyzer : this.modelAnalyzers) {
			LOGGER.info(SimulatorUtil.SEPARATOR + "\t" + analyzer.getName()
					+ ": " + analyzer.getDescription()
					+ SimulatorUtil.SEPARATOR);
			int failures = analyzer.analyze(this.eArchitecture);
			failureCount = failureCount + failures;
		}

		LOGGER.info(SimulatorUtil.DOUBLE_SEPARATOR + failureCount
				+ " adaptation causes have been identified."
				+ SimulatorUtil.DOUBLE_SEPARATOR);
		return failureCount;
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public void injectIssues() {
		if (this.injectors.size() == 0) {
			throw new RuntimeException("No injectors added to the simulator.");
		}

		this.runCount++;
		

		// initiate causes for adaptation
		LOGGER.info("\n\n\n**********************************************************************************************\n"
				+ "**********************************************************************************************\n"
				+ "\tSTART "
				+ this.runCount
				+ ". RUN OF INJECTION ISSUES (= RUN OF THE FEEDBACK LOOP) ..."
				+ "\n**********************************************************************************************\n"
				+ "**********************************************************************************************");
		this.injectionController.injectIssues(this.runCount);

		LOGGER.info("-----------------------------------------------------------------------------------------\n");

		// clear events
		ChangeEventQueue.INSTANCE.clearEvents();

		// return to let the feedback loop perform the adaptation and to observe
		// the change events due to adaptation
		LOGGER.info(SimulatorUtil.DOUBLE_SEPARATOR
				+ "\tFeedback loop should be run now ..."
				+ SimulatorUtil.DOUBLE_SEPARATOR + "\n");
	}

	/**
	 * Configures the logging of the simulator.
	 * 
	 * @param logLevel
	 *            the log level of the simulator.
	 * @param logFile
	 *            the name of the file that the logs will be stored. If this
	 *            parameter is <code>null</code> or the empty String, logging to
	 *            a file is disabled.
	 * @param logToConsole
	 *            if <code>true</code>, logs are printed to the console,
	 *            otherwise not.
	 */
	private void configLogging(Level logLevel, String logFile,
			boolean logToConsole) {

		// get the global logger to configure it
		Logger logger = Logger
				.getLogger("de.mdelab.morisia.comparch.simulator");
		logger.setLevel(logLevel);

		// Handlers are configured below. So no need for any parent handler.
		logger.setUseParentHandlers(false);

		// create a TXT formatter
		Formatter formatterTxt = new SimpleFormatter();
		formatterTxt = new java.util.logging.Formatter() {

			@Override
			public String format(LogRecord record) {
				StringBuffer buffer = new StringBuffer();
				buffer.append("[" + record.getLevel() + "] ");

				String loggerName = record.getLoggerName();
				String simpleLoggerName = loggerName.substring(loggerName
						.lastIndexOf(".") + 1);
				buffer.append(simpleLoggerName + ": ");
				buffer.append(record.getMessage());
				buffer.append("\n");
				return buffer.toString();
			}
		};

		if (logToConsole) {
			ConsoleHandler consoleHandler = new ConsoleHandler();
			consoleHandler.setFormatter(formatterTxt);
			consoleHandler.setLevel(logLevel);
			logger.addHandler(consoleHandler);
		}

		if (logFile != null && !logFile.equals("")) {
			try {
				FileHandler fileHandler = new FileHandler(logFile);
				fileHandler.setLevel(logLevel);
				fileHandler.setFormatter(formatterTxt);
				logger.addHandler(fileHandler);
			} catch (SecurityException e) {
				e.printStackTrace();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}

	}

}
