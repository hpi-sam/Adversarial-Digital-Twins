package de.mdelab.morisia.comparch;

import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.util.EcoreUtil;

import de.mdelab.morisia.comparch.impl.ComparchFactoryImpl;

/**
 * Factory for Comparch elements that automatically sets the unique identifier
 * of an element when the element is created.
 * 
 * @author thomas vogel
 *
 */
public class DefaultComparchFactoryImpl extends ComparchFactoryImpl {

	public static final DefaultComparchFactoryImpl eINSTANCE = new DefaultComparchFactoryImpl();

	public DefaultComparchFactoryImpl() {
		super();
	}

	private String genUUID(EObject object) {
		return EcoreUtil.generateUUID();
	}

	@Override
	public Architecture createArchitecture() {
		Architecture architecture = super.createArchitecture();
		architecture.setUid(this.genUUID(architecture));
		return architecture;
	}

	@Override
	public ComponentType createComponentType() {
		ComponentType componentType = super.createComponentType();
		componentType.setUid(this.genUUID(componentType));
		return componentType;
	}

	@Override
	public InterfaceType createInterfaceType() {
		InterfaceType interfaceType = super.createInterfaceType();
		interfaceType.setUid(this.genUUID(interfaceType));
		return interfaceType;
	}

	@Override
	public MethodSpecification createMethodSpecification() {
		MethodSpecification methodSpecification = super.createMethodSpecification();
		methodSpecification.setUid(this.genUUID(methodSpecification));
		return methodSpecification;
	}

	@Override
	public ParameterType createParameterType() {
		ParameterType parameterType = super.createParameterType();
		parameterType.setUid(this.genUUID(parameterType));
		return parameterType;
	}

	@Override
	public Tenant createTenant() {
		Tenant tenant = super.createTenant();
		tenant.setUid(this.genUUID(tenant));
		return tenant;
	}

	@Override
	public Component createComponent() {
		Component component = super.createComponent();
		component.setUid(this.genUUID(component));
		return component;
	}

	@Override
	public Parameter createParameter() {
		Parameter parameter = super.createParameter();
		parameter.setUid(this.genUUID(parameter));
		return parameter;
	}

	@Override
	public RequiredInterface createRequiredInterface() {
		RequiredInterface requiredInterface = super.createRequiredInterface();
		requiredInterface.setUid(this.genUUID(requiredInterface));
		return requiredInterface;
	}

	@Override
	public ProvidedInterface createProvidedInterface() {
		ProvidedInterface providedInterface = super.createProvidedInterface();
		providedInterface.setUid(this.genUUID(providedInterface));
		return providedInterface;
	}

	@Override
	public Connector createConnector() {
		Connector connector = super.createConnector();
		connector.setUid(this.genUUID(connector));
		return connector;
	}

	@Override
	public Failure createFailure() {
		Failure failure = super.createFailure();
		failure.setUid(this.genUUID(failure));
		return failure;
	}

	@Override
	public PerformanceStats createPerformanceStats() {
		PerformanceStats performanceStats = super.createPerformanceStats();
		performanceStats.setUid(this.genUUID(performanceStats));
		return performanceStats;
	}

	@Override
	public MonitoredProperty createMonitoredProperty() {
		MonitoredProperty monitoredProperty = super.createMonitoredProperty();
		monitoredProperty.setUid(this.genUUID(monitoredProperty));
		return monitoredProperty;
	}

	@Override
	public Annotations createAnnotations() {
		Annotations annotations = super.createAnnotations();
		return annotations;
	}

	@Override
	public CF1 createCF1() {
		CF1 cf1 = super.createCF1();
		return cf1;
	}

	@Override
	public CF2 createCF2() {
		CF2 cf2 = super.createCF2();
		return cf2;
	}

	@Override
	public CF3 createCF3() {
		CF3 cf3 = super.createCF3();
		return cf3;
	}

	@Override
	public CF4 createCF4() {
		CF4 cf4 = super.createCF4();
		return cf4;
	}
	
	@Override
	public CF5 createCF5() {
		CF5 cf5 = super.createCF5();
		return cf5;
	}

	@Override
	public RestartComponent createRestartComponent() {
		RestartComponent rs = super.createRestartComponent();
		return rs;
	}

	@Override
	public LwRedeployComponent createLwRedeployComponent() {
		LwRedeployComponent lrc = super.createLwRedeployComponent();
		return lrc;
	}

	@Override
	public HwRedeployComponent createHwRedeployComponent() {
		HwRedeployComponent hrc = super.createHwRedeployComponent();
		return hrc;
	}

	@Override
	public ReplaceComponent createReplaceComponent() {
		ReplaceComponent rc = super.createReplaceComponent();
		return rc;
	}

	@Override
	public RecreateConnector createRecreateConnector() {
		RecreateConnector rc = super.createRecreateConnector();
		return rc;
	}

}
