package de.mdelab.morisia.comparch.simulator;

import java.util.logging.Level;

import de.mdelab.morisia.comparch.Architecture;
import de.mdelab.morisia.comparch.simulator.impl.RandomInjectionStrategy;
import de.mdelab.morisia.comparch.simulator.mrubis.analyzer.adaptation.LifecycleAdaptationAnalyzer;
import de.mdelab.morisia.comparch.simulator.mrubis.analyzer.adaptation.MissingMonitoredPropertyAnalyzer;
import de.mdelab.morisia.comparch.simulator.mrubis.analyzer.adaptation.PipeAdaptationAnalyzer;
import de.mdelab.morisia.comparch.simulator.mrubis.analyzer.model.AttachedFailureAnalyzer;
import de.mdelab.morisia.comparch.simulator.mrubis.analyzer.model.AverageResponseTimeAnalyzer;
import de.mdelab.morisia.comparch.simulator.mrubis.analyzer.model.FilterOrderAnalyzer;
import de.mdelab.morisia.comparch.simulator.mrubis.analyzer.model.FilterTypeAnalyzer;
import de.mdelab.morisia.comparch.simulator.mrubis.analyzer.model.MinimalConfigurationAnalyzer;
import de.mdelab.morisia.comparch.simulator.mrubis.analyzer.model.ParameterAnalyzer;
import de.mdelab.morisia.comparch.simulator.mrubis.analyzer.model.PipeSizeAnalyzer;
import de.mdelab.morisia.comparch.simulator.mrubis.analyzer.model.ProvidedInterfaceAnalyzer;
import de.mdelab.morisia.comparch.simulator.mrubis.analyzer.model.RequiredInterfaceAnalyzer;
import de.mdelab.morisia.comparch.simulator.mrubis.injector.CF1_Injector;
import de.mdelab.morisia.comparch.simulator.mrubis.injector.CF2_Injector;
import de.mdelab.morisia.comparch.simulator.mrubis.injector.CF3_Injector;
import de.mdelab.morisia.comparch.simulator.mrubis.injector.CF4_Injector;
import de.mdelab.morisia.comparch.simulator.mrubis.injector.CF5_Injector;
import de.mdelab.morisia.comparch.simulator.mrubis.injector.NoCF2_Injector;
import de.mdelab.morisia.comparch.simulator.mrubis.injector.PI1_Injector;
import de.mdelab.morisia.comparch.simulator.mrubis.injector.PI2_Injector;
import de.mdelab.morisia.comparch.simulator.mrubis.injector.PI3_Injector;

/**
 * Factory to obtain an instance of the {@link ComparchSimulator}.
 * 
 * @author thomas vogel
 * @version 0.30
 *
 */
public class ComparchSimulatorFactory {

	/**
	 * Creates, if desired also configures, and returns an instance of the
	 * {@code ComparchSimulator}.
	 * 
	 * @param capability
	 *            the desired {@link Capability} that the simulator should
	 *            support. According to this capability, the simulator will be
	 *            configured with the required injectors and analyzers.
	 * @param eArchitecture
	 *            the {@link Architecture} element of the model (i.e., the root
	 *            element of the model).
	 * @param runs
	 *            number of runs the simulator should run. One run is performed
	 *            when invoking {@link ComparchSimulator#injectIssues()} which
	 *            injects issues into the model. Issues injected by one run of
	 *            the simulator should be handled by one run of the feedback
	 *            loop.
	 * @param logLevel
	 *            the log level of the simulator.
	 * @param logFile
	 *            the name of the file that the logs will be stored. If this
	 *            parameter is <code>null</code>, logging to a file is disabled.
	 * @param logToConsole
	 *            if <code>true</code>, logs are printed to the console,
	 *            otherwise not.
	 * @return an instance of the {@link ComparchSimulator} configured by the
	 *         given parameters.
	 */
	public ComparchSimulator createSimulator(Capability capability,
			Architecture eArchitecture, int runs, Level logLevel,
			String logFile, boolean logToConsole) {
		ComparchSimulator simulator = new ComparchSimulatorImpl(eArchitecture,
				runs, logLevel, logFile, logToConsole);

		switch (capability) {
		case SELF_REPAIR:
			this.configSelfHealing(simulator);
			break;
		case SELF_OPTIMIZATION:
			this.configSelfOptimization(simulator);
			break;
		case SELF_MANAGEMENT:
			this.configSelfManagement(simulator);
			break;
		case BLANK:
			// do nothing; requires adding analyzers and injectors by the user
			break;
		default:
			break;
		}

		InjectionStrategy defaultStrategy = new RandomInjectionStrategy(
				simulator.getSupportedIssueTypes(), eArchitecture);
		simulator.setInjectionStrategy(defaultStrategy);
		return simulator;
	}

	private void configSelfHealing(ComparchSimulator simulator) {
		// adaptation analyzer
		simulator.addAnalyzer(new LifecycleAdaptationAnalyzer());
		// model analyzer
		simulator.addAnalyzer(new ProvidedInterfaceAnalyzer());
		simulator.addAnalyzer(new RequiredInterfaceAnalyzer());
		simulator.addAnalyzer(new ParameterAnalyzer());
		simulator.addAnalyzer(new MinimalConfigurationAnalyzer());
		simulator.addAnalyzer(new AttachedFailureAnalyzer(5));
		simulator.addAnalyzer(new PipeSizeAnalyzer(4));
		simulator.addAnalyzer(new FilterTypeAnalyzer());

		// injectors
		simulator.addInjector(new CF1_Injector());
		simulator.addInjector(new NoCF2_Injector(5));
		simulator.addInjector(new CF2_Injector(5));
		simulator.addInjector(new CF3_Injector());
		simulator.addInjector(new CF4_Injector());
		simulator.addInjector(new CF5_Injector());

	}

	private void configSelfOptimization(ComparchSimulator simulator) {

		int performanceMonitoringIntervall = 50;
		int responseTimeGoal = 1000;
		int responseTimeUpperThreshold = 1150;
		int responseTimeLowerThreshold = 850;

		simulator.addAnalyzer(new LifecycleAdaptationAnalyzer());
		simulator.addAnalyzer(new MissingMonitoredPropertyAnalyzer());
		simulator.addAnalyzer(new PipeAdaptationAnalyzer(
				performanceMonitoringIntervall));
		// model analyzer
		simulator.addAnalyzer(new ProvidedInterfaceAnalyzer());
		simulator.addAnalyzer(new RequiredInterfaceAnalyzer());
		simulator.addAnalyzer(new ParameterAnalyzer());
		simulator.addAnalyzer(new MinimalConfigurationAnalyzer());
		simulator.addAnalyzer(new PipeSizeAnalyzer(4));
		simulator.addAnalyzer(new FilterTypeAnalyzer());
		simulator.addAnalyzer(new FilterOrderAnalyzer());
		simulator.addAnalyzer(new AverageResponseTimeAnalyzer(
				responseTimeUpperThreshold, responseTimeLowerThreshold,
				responseTimeGoal));

		// injectors
		simulator.addInjector(new PI1_Injector());
		simulator.addInjector(new PI2_Injector(performanceMonitoringIntervall,
				responseTimeUpperThreshold));
		simulator.addInjector(new PI3_Injector(performanceMonitoringIntervall,
				responseTimeLowerThreshold));
	}

	private void configSelfManagement(ComparchSimulator simulator) {

		int performanceMonitoringIntervall = 50;
		int responseTimeGoal = 1000;
		int responseTimeUpperThreshold = 1150;
		int responseTimeLowerThreshold = 850;

		// adaptation analyzer
		simulator.addAnalyzer(new LifecycleAdaptationAnalyzer());
		simulator.addAnalyzer(new MissingMonitoredPropertyAnalyzer());
		simulator.addAnalyzer(new PipeAdaptationAnalyzer(
				performanceMonitoringIntervall));
		// model analyzer
		simulator.addAnalyzer(new ProvidedInterfaceAnalyzer());
		simulator.addAnalyzer(new RequiredInterfaceAnalyzer());
		simulator.addAnalyzer(new ParameterAnalyzer());
		simulator.addAnalyzer(new MinimalConfigurationAnalyzer());
		simulator.addAnalyzer(new AttachedFailureAnalyzer(5));
		simulator.addAnalyzer(new PipeSizeAnalyzer(4));
		simulator.addAnalyzer(new FilterTypeAnalyzer());
		simulator.addAnalyzer(new FilterOrderAnalyzer());
		simulator.addAnalyzer(new AverageResponseTimeAnalyzer(
				responseTimeUpperThreshold, responseTimeLowerThreshold,
				responseTimeGoal));

		// injectors
		simulator.addInjector(new CF1_Injector());
		simulator.addInjector(new NoCF2_Injector(5));
		simulator.addInjector(new CF2_Injector(5));
		simulator.addInjector(new CF3_Injector());
		simulator.addInjector(new CF4_Injector());
		simulator.addInjector(new CF5_Injector());
		simulator.addInjector(new PI1_Injector());
		simulator.addInjector(new PI2_Injector(performanceMonitoringIntervall,
				responseTimeUpperThreshold));
		simulator.addInjector(new PI3_Injector(performanceMonitoringIntervall,
				responseTimeLowerThreshold));
	}

}
