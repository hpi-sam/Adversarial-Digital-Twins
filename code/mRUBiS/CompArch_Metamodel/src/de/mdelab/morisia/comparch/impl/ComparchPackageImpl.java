/**
 */
package de.mdelab.morisia.comparch.impl;

import de.mdelab.morisia.comparch.AddReplica;
import de.mdelab.morisia.comparch.Annotations;
import org.eclipse.emf.ecore.EAttribute;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EEnum;
import org.eclipse.emf.ecore.EPackage;
import org.eclipse.emf.ecore.EReference;
import org.eclipse.emf.ecore.impl.EPackageImpl;

import de.mdelab.morisia.comparch.ArchitecturalElement;
import de.mdelab.morisia.comparch.Architecture;
import de.mdelab.morisia.comparch.Cloud;
import de.mdelab.morisia.comparch.ComparchFactory;
import de.mdelab.morisia.comparch.ComparchPackage;
import de.mdelab.morisia.comparch.Component;
import de.mdelab.morisia.comparch.ComponentLifeCycle;
import de.mdelab.morisia.comparch.ComponentType;
import de.mdelab.morisia.comparch.Connector;
import de.mdelab.morisia.comparch.Failure;
import de.mdelab.morisia.comparch.HwRedeployComponent;
import de.mdelab.morisia.comparch.Interface;
import de.mdelab.morisia.comparch.InterfaceType;
import de.mdelab.morisia.comparch.Issue;
import de.mdelab.morisia.comparch.LwRedeployComponent;
import de.mdelab.morisia.comparch.MethodSpecification;
import de.mdelab.morisia.comparch.MonitoredProperty;
import de.mdelab.morisia.comparch.MoveReplica;
import de.mdelab.morisia.comparch.Parameter;
import de.mdelab.morisia.comparch.ParameterType;
import de.mdelab.morisia.comparch.PerformanceStats;
import de.mdelab.morisia.comparch.ProvidedInterface;
import de.mdelab.morisia.comparch.RecreateConnector;
import de.mdelab.morisia.comparch.RemoveReplica;
import de.mdelab.morisia.comparch.ReplaceComponent;
import de.mdelab.morisia.comparch.Replica;
import de.mdelab.morisia.comparch.RequiredInterface;
import de.mdelab.morisia.comparch.RestartComponent;
import de.mdelab.morisia.comparch.Rule;
import de.mdelab.morisia.comparch.Server;
import de.mdelab.morisia.comparch.Tenant;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model <b>Package</b>.
 * <!-- end-user-doc -->
 * @generated
 */
public class ComparchPackageImpl extends EPackageImpl implements ComparchPackage {
	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass architectureEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass architecturalElementEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass componentTypeEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass interfaceTypeEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass parameterTypeEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass componentEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass replicaEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass parameterEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass methodSpecificationEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass interfaceEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass requiredInterfaceEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass providedInterfaceEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass connectorEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass failureEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass performanceStatsEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass monitoredPropertyEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass tenantEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass annotationsEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass issueEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass cfEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass cf1EClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass cf2EClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass cf3EClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass cf5EClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass ruleEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass moveReplicaEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass addReplicaEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass removeReplicaEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass hwRedeployComponentEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass lwRedeployComponentEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass restartComponentEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass replaceComponentEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass cf4EClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass recreateConnectorEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass serverEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EClass cloudEClass = null;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private EEnum componentLifeCycleEEnum = null;

	/**
	 * Creates an instance of the model <b>Package</b>, registered with
	 * {@link org.eclipse.emf.ecore.EPackage.Registry EPackage.Registry} by the package
	 * package URI value.
	 * <p>Note: the correct way to create the package is via the static
	 * factory method {@link #init init()}, which also performs
	 * initialization of the package, or returns the registered package,
	 * if one already exists.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see org.eclipse.emf.ecore.EPackage.Registry
	 * @see de.mdelab.morisia.comparch.ComparchPackage#eNS_URI
	 * @see #init()
	 * @generated
	 */
	private ComparchPackageImpl() {
		super(eNS_URI, ComparchFactory.eINSTANCE);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private static boolean isInited = false;

	/**
	 * Creates, registers, and initializes the <b>Package</b> for this model, and for any others upon which it depends.
	 * 
	 * <p>This method is used to initialize {@link ComparchPackage#eINSTANCE} when that field is accessed.
	 * Clients should not invoke it directly. Instead, they should simply access that field to obtain the package.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #eNS_URI
	 * @see #createPackageContents()
	 * @see #initializePackageContents()
	 * @generated
	 */
	public static ComparchPackage init() {
		if (isInited) return (ComparchPackage)EPackage.Registry.INSTANCE.getEPackage(ComparchPackage.eNS_URI);

		// Obtain or create and register package
		ComparchPackageImpl theComparchPackage = (ComparchPackageImpl)(EPackage.Registry.INSTANCE.get(eNS_URI) instanceof ComparchPackageImpl ? EPackage.Registry.INSTANCE.get(eNS_URI) : new ComparchPackageImpl());

		isInited = true;

		// Create package meta-data objects
		theComparchPackage.createPackageContents();

		// Initialize created meta-data
		theComparchPackage.initializePackageContents();

		// Mark meta-data to indicate it can't be changed
		theComparchPackage.freeze();

  
		// Update the registry and return the package
		EPackage.Registry.INSTANCE.put(ComparchPackage.eNS_URI, theComparchPackage);
		return theComparchPackage;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getArchitecture() {
		return architectureEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getArchitecture_Name() {
		return (EAttribute)architectureEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getArchitecture_ComponentTypes() {
		return (EReference)architectureEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getArchitecture_InterfaceTypes() {
		return (EReference)architectureEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getArchitecture_Tenants() {
		return (EReference)architectureEClass.getEStructuralFeatures().get(3);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getArchitecture_Cloud() {
		return (EReference)architectureEClass.getEStructuralFeatures().get(4);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getArchitecture_Annotations() {
		return (EReference)architectureEClass.getEStructuralFeatures().get(5);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getArchitecturalElement() {
		return architecturalElementEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getArchitecturalElement_Uid() {
		return (EAttribute)architecturalElementEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getArchitecturalElement_Criticality() {
		return (EAttribute)architecturalElementEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getArchitecturalElement_Importance() {
		return (EAttribute)architecturalElementEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getArchitecturalElement_ADT() {
		return (EAttribute)architecturalElementEClass.getEStructuralFeatures().get(3);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getArchitecturalElement_Efficiency() {
		return (EAttribute)architecturalElementEClass.getEStructuralFeatures().get(4);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getArchitecturalElement_Performance() {
		return (EAttribute)architecturalElementEClass.getEStructuralFeatures().get(5);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getArchitecturalElement_Request() {
		return (EAttribute)architecturalElementEClass.getEStructuralFeatures().get(6);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getComponentType() {
		return componentTypeEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getComponentType_Name() {
		return (EAttribute)componentTypeEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getComponentType_Architecture() {
		return (EReference)componentTypeEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getComponentType_RequiredInterfaceTypes() {
		return (EReference)componentTypeEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getComponentType_ProvidedInterfaceTypes() {
		return (EReference)componentTypeEClass.getEStructuralFeatures().get(3);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getComponentType_ParameterTypes() {
		return (EReference)componentTypeEClass.getEStructuralFeatures().get(4);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getComponentType_Instances() {
		return (EReference)componentTypeEClass.getEStructuralFeatures().get(5);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getComponentType_Reliability() {
		return (EAttribute)componentTypeEClass.getEStructuralFeatures().get(6);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getComponentType_SatPoint() {
		return (EAttribute)componentTypeEClass.getEStructuralFeatures().get(7);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getComponentType_PerformanceMax() {
		return (EAttribute)componentTypeEClass.getEStructuralFeatures().get(8);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getComponentType_ReplicaCost() {
		return (EAttribute)componentTypeEClass.getEStructuralFeatures().get(9);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getComponentType_ReplicaADT() {
		return (EAttribute)componentTypeEClass.getEStructuralFeatures().get(10);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getComponentType_CpuFactor() {
		return (EAttribute)componentTypeEClass.getEStructuralFeatures().get(11);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getComponentType_MemoryFactor() {
		return (EAttribute)componentTypeEClass.getEStructuralFeatures().get(12);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getComponentType_StorageFactor() {
		return (EAttribute)componentTypeEClass.getEStructuralFeatures().get(13);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getInterfaceType() {
		return interfaceTypeEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getInterfaceType_FullyQualifiedName() {
		return (EAttribute)interfaceTypeEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getInterfaceType_Architecture() {
		return (EReference)interfaceTypeEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getInterfaceType_RequiringComponentTypes() {
		return (EReference)interfaceTypeEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getInterfaceType_ProvidingComponentTypes() {
		return (EReference)interfaceTypeEClass.getEStructuralFeatures().get(3);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getInterfaceType_MethodSpecification() {
		return (EReference)interfaceTypeEClass.getEStructuralFeatures().get(4);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getInterfaceType_Instances() {
		return (EReference)interfaceTypeEClass.getEStructuralFeatures().get(5);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getParameterType() {
		return parameterTypeEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getParameterType_Name() {
		return (EAttribute)parameterTypeEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getParameterType_Type() {
		return (EAttribute)parameterTypeEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getParameterType_DefaultValue() {
		return (EAttribute)parameterTypeEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getParameterType_ComponentType() {
		return (EReference)parameterTypeEClass.getEStructuralFeatures().get(3);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getParameterType_Instances() {
		return (EReference)parameterTypeEClass.getEStructuralFeatures().get(4);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getComponent() {
		return componentEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getComponent_Parameters() {
		return (EReference)componentEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getComponent_RequiredInterfaces() {
		return (EReference)componentEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getComponent_State() {
		return (EAttribute)componentEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getComponent_ProvidedInterfaces() {
		return (EReference)componentEClass.getEStructuralFeatures().get(3);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getComponent_Type() {
		return (EReference)componentEClass.getEStructuralFeatures().get(4);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getComponent_MonitoredProperties() {
		return (EReference)componentEClass.getEStructuralFeatures().get(5);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getComponent_Tenant() {
		return (EReference)componentEClass.getEStructuralFeatures().get(6);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getComponent_Issues() {
		return (EReference)componentEClass.getEStructuralFeatures().get(7);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getComponent_InUseReplica() {
		return (EAttribute)componentEClass.getEStructuralFeatures().get(8);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getComponent_HealthStatus() {
		return (EAttribute)componentEClass.getEStructuralFeatures().get(9);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getComponent_PerformanceOld() {
		return (EAttribute)componentEClass.getEStructuralFeatures().get(10);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getComponent_OriginalRequest() {
		return (EAttribute)componentEClass.getEStructuralFeatures().get(11);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getComponent_DeployedOn() {
		return (EReference)componentEClass.getEStructuralFeatures().get(12);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getComponent_Replica() {
		return (EReference)componentEClass.getEStructuralFeatures().get(13);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getReplica() {
		return replicaEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getReplica_DeployedOn() {
		return (EReference)replicaEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getReplica_Cost() {
		return (EAttribute)replicaEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getReplica_InUse() {
		return (EAttribute)replicaEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getReplica_CPURequirement() {
		return (EAttribute)replicaEClass.getEStructuralFeatures().get(3);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getReplica_MemoryRequirement() {
		return (EAttribute)replicaEClass.getEStructuralFeatures().get(4);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getReplica_StorageRequirement() {
		return (EAttribute)replicaEClass.getEStructuralFeatures().get(5);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getReplica_Component() {
		return (EReference)replicaEClass.getEStructuralFeatures().get(6);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getParameter() {
		return parameterEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getParameter_Value() {
		return (EAttribute)parameterEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getParameter_LazyAdaptation() {
		return (EAttribute)parameterEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getParameter_Component() {
		return (EReference)parameterEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getParameter_Type() {
		return (EReference)parameterEClass.getEStructuralFeatures().get(3);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getMethodSpecification() {
		return methodSpecificationEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getMethodSpecification_Signature() {
		return (EAttribute)methodSpecificationEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getMethodSpecification_InterfaceType() {
		return (EReference)methodSpecificationEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getInterface() {
		return interfaceEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getInterface_Type() {
		return (EReference)interfaceEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getRequiredInterface() {
		return requiredInterfaceEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getRequiredInterface_Component() {
		return (EReference)requiredInterfaceEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getRequiredInterface_Connector() {
		return (EReference)requiredInterfaceEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getProvidedInterface() {
		return providedInterfaceEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getProvidedInterface_Component() {
		return (EReference)providedInterfaceEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getProvidedInterface_Connectors() {
		return (EReference)providedInterfaceEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getProvidedInterface_Failures() {
		return (EReference)providedInterfaceEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getProvidedInterface_PerformanceStats() {
		return (EReference)providedInterfaceEClass.getEStructuralFeatures().get(3);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getConnector() {
		return connectorEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getConnector_LazyAdaptation() {
		return (EAttribute)connectorEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getConnector_Source() {
		return (EReference)connectorEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getConnector_Target() {
		return (EReference)connectorEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getFailure() {
		return failureEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getFailure_Name() {
		return (EAttribute)failureEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getFailure_Message() {
		return (EAttribute)failureEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getFailure_Interface() {
		return (EReference)failureEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getFailure_Method() {
		return (EReference)failureEClass.getEStructuralFeatures().get(3);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getPerformanceStats() {
		return performanceStatsEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getPerformanceStats_MinTime() {
		return (EAttribute)performanceStatsEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getPerformanceStats_MaxTime() {
		return (EAttribute)performanceStatsEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getPerformanceStats_TotalTime() {
		return (EAttribute)performanceStatsEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getPerformanceStats_InvocationCount() {
		return (EAttribute)performanceStatsEClass.getEStructuralFeatures().get(3);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getPerformanceStats_Interface() {
		return (EReference)performanceStatsEClass.getEStructuralFeatures().get(4);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getPerformanceStats_Method() {
		return (EReference)performanceStatsEClass.getEStructuralFeatures().get(5);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getMonitoredProperty() {
		return monitoredPropertyEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getMonitoredProperty_Name() {
		return (EAttribute)monitoredPropertyEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getMonitoredProperty_Value() {
		return (EAttribute)monitoredPropertyEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getMonitoredProperty_Type() {
		return (EAttribute)monitoredPropertyEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getMonitoredProperty_Unit() {
		return (EAttribute)monitoredPropertyEClass.getEStructuralFeatures().get(3);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getMonitoredProperty_Description() {
		return (EAttribute)monitoredPropertyEClass.getEStructuralFeatures().get(4);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getMonitoredProperty_Component() {
		return (EReference)monitoredPropertyEClass.getEStructuralFeatures().get(5);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getMonitoredProperty_ConcernedElements() {
		return (EReference)monitoredPropertyEClass.getEStructuralFeatures().get(6);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getTenant() {
		return tenantEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getTenant_Name() {
		return (EAttribute)tenantEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getTenant_Architecture() {
		return (EReference)tenantEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getTenant_Components() {
		return (EReference)tenantEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getTenant_ReplicaFactor() {
		return (EAttribute)tenantEClass.getEStructuralFeatures().get(3);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getAnnotations() {
		return annotationsEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getAnnotations_Architecture() {
		return (EReference)annotationsEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getAnnotations_Issues() {
		return (EReference)annotationsEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getAnnotations_Rules() {
		return (EReference)annotationsEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getIssue() {
		return issueEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getIssue_UtilityDrop() {
		return (EAttribute)issueEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getIssue_AffectedComponent() {
		return (EReference)issueEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getIssue_Annotations() {
		return (EReference)issueEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getIssue_HandledBy() {
		return (EReference)issueEClass.getEStructuralFeatures().get(3);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getCF() {
		return cfEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getCF1() {
		return cf1EClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getCF2() {
		return cf2EClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getCF3() {
		return cf3EClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getCF5() {
		return cf5EClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getRule() {
		return ruleEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getRule_Costs() {
		return (EAttribute)ruleEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getRule_Annotations() {
		return (EReference)ruleEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getRule_Handles() {
		return (EReference)ruleEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getRule_UtilityIncrease() {
		return (EAttribute)ruleEClass.getEStructuralFeatures().get(3);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getMoveReplica() {
		return moveReplicaEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getAddReplica() {
		return addReplicaEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getRemoveReplica() {
		return removeReplicaEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getHwRedeployComponent() {
		return hwRedeployComponentEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getLwRedeployComponent() {
		return lwRedeployComponentEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getRestartComponent() {
		return restartComponentEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getReplaceComponent() {
		return replaceComponentEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getReplaceComponent_AlternativeComponentType() {
		return (EReference)replaceComponentEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getCF4() {
		return cf4EClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getRecreateConnector() {
		return recreateConnectorEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getServer() {
		return serverEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getServer_CPUCapacity() {
		return (EAttribute)serverEClass.getEStructuralFeatures().get(4);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getServer_MemoryCapacity() {
		return (EAttribute)serverEClass.getEStructuralFeatures().get(5);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getServer_StorageCapacity() {
		return (EAttribute)serverEClass.getEStructuralFeatures().get(6);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getServer_CPUConsumption() {
		return (EAttribute)serverEClass.getEStructuralFeatures().get(7);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getServer_MemoryConsumption() {
		return (EAttribute)serverEClass.getEStructuralFeatures().get(8);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getServer_StorageConsumption() {
		return (EAttribute)serverEClass.getEStructuralFeatures().get(9);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getServer_Cost() {
		return (EAttribute)serverEClass.getEStructuralFeatures().get(10);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getServer_CpuFactor() {
		return (EAttribute)serverEClass.getEStructuralFeatures().get(11);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getServer_MemoryFactor() {
		return (EAttribute)serverEClass.getEStructuralFeatures().get(12);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getServer_StorageFactor() {
		return (EAttribute)serverEClass.getEStructuralFeatures().get(13);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getServer_Cloud() {
		return (EReference)serverEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getServer_Name() {
		return (EAttribute)serverEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getServer_DeployedReplicas() {
		return (EReference)serverEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getServer_DeployedComponent() {
		return (EReference)serverEClass.getEStructuralFeatures().get(3);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EClass getCloud() {
		return cloudEClass;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getCloud_Servers() {
		return (EReference)cloudEClass.getEStructuralFeatures().get(0);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EReference getCloud_Architecture() {
		return (EReference)cloudEClass.getEStructuralFeatures().get(1);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EAttribute getCloud_Name() {
		return (EAttribute)cloudEClass.getEStructuralFeatures().get(2);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EEnum getComponentLifeCycle() {
		return componentLifeCycleEEnum;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ComparchFactory getComparchFactory() {
		return (ComparchFactory)getEFactoryInstance();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private boolean isCreated = false;

	/**
	 * Creates the meta-model objects for the package.  This method is
	 * guarded to have no affect on any invocation but its first.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void createPackageContents() {
		if (isCreated) return;
		isCreated = true;

		// Create classes and their features
		cloudEClass = createEClass(CLOUD);
		createEReference(cloudEClass, CLOUD__SERVERS);
		createEReference(cloudEClass, CLOUD__ARCHITECTURE);
		createEAttribute(cloudEClass, CLOUD__NAME);

		serverEClass = createEClass(SERVER);
		createEReference(serverEClass, SERVER__CLOUD);
		createEAttribute(serverEClass, SERVER__NAME);
		createEReference(serverEClass, SERVER__DEPLOYED_REPLICAS);
		createEReference(serverEClass, SERVER__DEPLOYED_COMPONENT);
		createEAttribute(serverEClass, SERVER__CPU_CAPACITY);
		createEAttribute(serverEClass, SERVER__MEMORY_CAPACITY);
		createEAttribute(serverEClass, SERVER__STORAGE_CAPACITY);
		createEAttribute(serverEClass, SERVER__CPU_CONSUMPTION);
		createEAttribute(serverEClass, SERVER__MEMORY_CONSUMPTION);
		createEAttribute(serverEClass, SERVER__STORAGE_CONSUMPTION);
		createEAttribute(serverEClass, SERVER__COST);
		createEAttribute(serverEClass, SERVER__CPU_FACTOR);
		createEAttribute(serverEClass, SERVER__MEMORY_FACTOR);
		createEAttribute(serverEClass, SERVER__STORAGE_FACTOR);

		architectureEClass = createEClass(ARCHITECTURE);
		createEAttribute(architectureEClass, ARCHITECTURE__NAME);
		createEReference(architectureEClass, ARCHITECTURE__COMPONENT_TYPES);
		createEReference(architectureEClass, ARCHITECTURE__INTERFACE_TYPES);
		createEReference(architectureEClass, ARCHITECTURE__TENANTS);
		createEReference(architectureEClass, ARCHITECTURE__CLOUD);
		createEReference(architectureEClass, ARCHITECTURE__ANNOTATIONS);

		architecturalElementEClass = createEClass(ARCHITECTURAL_ELEMENT);
		createEAttribute(architecturalElementEClass, ARCHITECTURAL_ELEMENT__UID);
		createEAttribute(architecturalElementEClass, ARCHITECTURAL_ELEMENT__CRITICALITY);
		createEAttribute(architecturalElementEClass, ARCHITECTURAL_ELEMENT__IMPORTANCE);
		createEAttribute(architecturalElementEClass, ARCHITECTURAL_ELEMENT__ADT);
		createEAttribute(architecturalElementEClass, ARCHITECTURAL_ELEMENT__EFFICIENCY);
		createEAttribute(architecturalElementEClass, ARCHITECTURAL_ELEMENT__PERFORMANCE);
		createEAttribute(architecturalElementEClass, ARCHITECTURAL_ELEMENT__REQUEST);

		componentTypeEClass = createEClass(COMPONENT_TYPE);
		createEAttribute(componentTypeEClass, COMPONENT_TYPE__NAME);
		createEReference(componentTypeEClass, COMPONENT_TYPE__ARCHITECTURE);
		createEReference(componentTypeEClass, COMPONENT_TYPE__REQUIRED_INTERFACE_TYPES);
		createEReference(componentTypeEClass, COMPONENT_TYPE__PROVIDED_INTERFACE_TYPES);
		createEReference(componentTypeEClass, COMPONENT_TYPE__PARAMETER_TYPES);
		createEReference(componentTypeEClass, COMPONENT_TYPE__INSTANCES);
		createEAttribute(componentTypeEClass, COMPONENT_TYPE__RELIABILITY);
		createEAttribute(componentTypeEClass, COMPONENT_TYPE__SAT_POINT);
		createEAttribute(componentTypeEClass, COMPONENT_TYPE__PERFORMANCE_MAX);
		createEAttribute(componentTypeEClass, COMPONENT_TYPE__REPLICA_COST);
		createEAttribute(componentTypeEClass, COMPONENT_TYPE__REPLICA_ADT);
		createEAttribute(componentTypeEClass, COMPONENT_TYPE__CPU_FACTOR);
		createEAttribute(componentTypeEClass, COMPONENT_TYPE__MEMORY_FACTOR);
		createEAttribute(componentTypeEClass, COMPONENT_TYPE__STORAGE_FACTOR);

		interfaceTypeEClass = createEClass(INTERFACE_TYPE);
		createEAttribute(interfaceTypeEClass, INTERFACE_TYPE__FULLY_QUALIFIED_NAME);
		createEReference(interfaceTypeEClass, INTERFACE_TYPE__ARCHITECTURE);
		createEReference(interfaceTypeEClass, INTERFACE_TYPE__REQUIRING_COMPONENT_TYPES);
		createEReference(interfaceTypeEClass, INTERFACE_TYPE__PROVIDING_COMPONENT_TYPES);
		createEReference(interfaceTypeEClass, INTERFACE_TYPE__METHOD_SPECIFICATION);
		createEReference(interfaceTypeEClass, INTERFACE_TYPE__INSTANCES);

		parameterTypeEClass = createEClass(PARAMETER_TYPE);
		createEAttribute(parameterTypeEClass, PARAMETER_TYPE__NAME);
		createEAttribute(parameterTypeEClass, PARAMETER_TYPE__TYPE);
		createEAttribute(parameterTypeEClass, PARAMETER_TYPE__DEFAULT_VALUE);
		createEReference(parameterTypeEClass, PARAMETER_TYPE__COMPONENT_TYPE);
		createEReference(parameterTypeEClass, PARAMETER_TYPE__INSTANCES);

		componentEClass = createEClass(COMPONENT);
		createEReference(componentEClass, COMPONENT__PARAMETERS);
		createEReference(componentEClass, COMPONENT__REQUIRED_INTERFACES);
		createEAttribute(componentEClass, COMPONENT__STATE);
		createEReference(componentEClass, COMPONENT__PROVIDED_INTERFACES);
		createEReference(componentEClass, COMPONENT__TYPE);
		createEReference(componentEClass, COMPONENT__MONITORED_PROPERTIES);
		createEReference(componentEClass, COMPONENT__TENANT);
		createEReference(componentEClass, COMPONENT__ISSUES);
		createEAttribute(componentEClass, COMPONENT__IN_USE_REPLICA);
		createEAttribute(componentEClass, COMPONENT__HEALTH_STATUS);
		createEAttribute(componentEClass, COMPONENT__PERFORMANCE_OLD);
		createEAttribute(componentEClass, COMPONENT__ORIGINAL_REQUEST);
		createEReference(componentEClass, COMPONENT__DEPLOYED_ON);
		createEReference(componentEClass, COMPONENT__REPLICA);

		replicaEClass = createEClass(REPLICA);
		createEReference(replicaEClass, REPLICA__DEPLOYED_ON);
		createEAttribute(replicaEClass, REPLICA__COST);
		createEAttribute(replicaEClass, REPLICA__IN_USE);
		createEAttribute(replicaEClass, REPLICA__CPU_REQUIREMENT);
		createEAttribute(replicaEClass, REPLICA__MEMORY_REQUIREMENT);
		createEAttribute(replicaEClass, REPLICA__STORAGE_REQUIREMENT);
		createEReference(replicaEClass, REPLICA__COMPONENT);

		parameterEClass = createEClass(PARAMETER);
		createEAttribute(parameterEClass, PARAMETER__VALUE);
		createEAttribute(parameterEClass, PARAMETER__LAZY_ADAPTATION);
		createEReference(parameterEClass, PARAMETER__COMPONENT);
		createEReference(parameterEClass, PARAMETER__TYPE);

		methodSpecificationEClass = createEClass(METHOD_SPECIFICATION);
		createEAttribute(methodSpecificationEClass, METHOD_SPECIFICATION__SIGNATURE);
		createEReference(methodSpecificationEClass, METHOD_SPECIFICATION__INTERFACE_TYPE);

		interfaceEClass = createEClass(INTERFACE);
		createEReference(interfaceEClass, INTERFACE__TYPE);

		requiredInterfaceEClass = createEClass(REQUIRED_INTERFACE);
		createEReference(requiredInterfaceEClass, REQUIRED_INTERFACE__COMPONENT);
		createEReference(requiredInterfaceEClass, REQUIRED_INTERFACE__CONNECTOR);

		providedInterfaceEClass = createEClass(PROVIDED_INTERFACE);
		createEReference(providedInterfaceEClass, PROVIDED_INTERFACE__COMPONENT);
		createEReference(providedInterfaceEClass, PROVIDED_INTERFACE__CONNECTORS);
		createEReference(providedInterfaceEClass, PROVIDED_INTERFACE__FAILURES);
		createEReference(providedInterfaceEClass, PROVIDED_INTERFACE__PERFORMANCE_STATS);

		connectorEClass = createEClass(CONNECTOR);
		createEAttribute(connectorEClass, CONNECTOR__LAZY_ADAPTATION);
		createEReference(connectorEClass, CONNECTOR__SOURCE);
		createEReference(connectorEClass, CONNECTOR__TARGET);

		failureEClass = createEClass(FAILURE);
		createEAttribute(failureEClass, FAILURE__NAME);
		createEAttribute(failureEClass, FAILURE__MESSAGE);
		createEReference(failureEClass, FAILURE__INTERFACE);
		createEReference(failureEClass, FAILURE__METHOD);

		performanceStatsEClass = createEClass(PERFORMANCE_STATS);
		createEAttribute(performanceStatsEClass, PERFORMANCE_STATS__MIN_TIME);
		createEAttribute(performanceStatsEClass, PERFORMANCE_STATS__MAX_TIME);
		createEAttribute(performanceStatsEClass, PERFORMANCE_STATS__TOTAL_TIME);
		createEAttribute(performanceStatsEClass, PERFORMANCE_STATS__INVOCATION_COUNT);
		createEReference(performanceStatsEClass, PERFORMANCE_STATS__INTERFACE);
		createEReference(performanceStatsEClass, PERFORMANCE_STATS__METHOD);

		monitoredPropertyEClass = createEClass(MONITORED_PROPERTY);
		createEAttribute(monitoredPropertyEClass, MONITORED_PROPERTY__NAME);
		createEAttribute(monitoredPropertyEClass, MONITORED_PROPERTY__VALUE);
		createEAttribute(monitoredPropertyEClass, MONITORED_PROPERTY__TYPE);
		createEAttribute(monitoredPropertyEClass, MONITORED_PROPERTY__UNIT);
		createEAttribute(monitoredPropertyEClass, MONITORED_PROPERTY__DESCRIPTION);
		createEReference(monitoredPropertyEClass, MONITORED_PROPERTY__COMPONENT);
		createEReference(monitoredPropertyEClass, MONITORED_PROPERTY__CONCERNED_ELEMENTS);

		tenantEClass = createEClass(TENANT);
		createEAttribute(tenantEClass, TENANT__NAME);
		createEReference(tenantEClass, TENANT__ARCHITECTURE);
		createEReference(tenantEClass, TENANT__COMPONENTS);
		createEAttribute(tenantEClass, TENANT__REPLICA_FACTOR);

		annotationsEClass = createEClass(ANNOTATIONS);
		createEReference(annotationsEClass, ANNOTATIONS__ARCHITECTURE);
		createEReference(annotationsEClass, ANNOTATIONS__ISSUES);
		createEReference(annotationsEClass, ANNOTATIONS__RULES);

		issueEClass = createEClass(ISSUE);
		createEAttribute(issueEClass, ISSUE__UTILITY_DROP);
		createEReference(issueEClass, ISSUE__AFFECTED_COMPONENT);
		createEReference(issueEClass, ISSUE__ANNOTATIONS);
		createEReference(issueEClass, ISSUE__HANDLED_BY);

		cfEClass = createEClass(CF);

		cf1EClass = createEClass(CF1);

		cf2EClass = createEClass(CF2);

		cf3EClass = createEClass(CF3);

		cf5EClass = createEClass(CF5);

		ruleEClass = createEClass(RULE);
		createEAttribute(ruleEClass, RULE__COSTS);
		createEReference(ruleEClass, RULE__ANNOTATIONS);
		createEReference(ruleEClass, RULE__HANDLES);
		createEAttribute(ruleEClass, RULE__UTILITY_INCREASE);

		moveReplicaEClass = createEClass(MOVE_REPLICA);

		addReplicaEClass = createEClass(ADD_REPLICA);

		removeReplicaEClass = createEClass(REMOVE_REPLICA);

		hwRedeployComponentEClass = createEClass(HW_REDEPLOY_COMPONENT);

		lwRedeployComponentEClass = createEClass(LW_REDEPLOY_COMPONENT);

		restartComponentEClass = createEClass(RESTART_COMPONENT);

		replaceComponentEClass = createEClass(REPLACE_COMPONENT);
		createEReference(replaceComponentEClass, REPLACE_COMPONENT__ALTERNATIVE_COMPONENT_TYPE);

		cf4EClass = createEClass(CF4);

		recreateConnectorEClass = createEClass(RECREATE_CONNECTOR);

		// Create enums
		componentLifeCycleEEnum = createEEnum(COMPONENT_LIFE_CYCLE);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private boolean isInitialized = false;

	/**
	 * Complete the initialization of the package and its meta-model.  This
	 * method is guarded to have no affect on any invocation but its first.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void initializePackageContents() {
		if (isInitialized) return;
		isInitialized = true;

		// Initialize package
		setName(eNAME);
		setNsPrefix(eNS_PREFIX);
		setNsURI(eNS_URI);

		// Create type parameters

		// Set bounds for type parameters

		// Add supertypes to classes
		cloudEClass.getESuperTypes().add(this.getArchitecturalElement());
		serverEClass.getESuperTypes().add(this.getArchitecturalElement());
		architectureEClass.getESuperTypes().add(this.getArchitecturalElement());
		componentTypeEClass.getESuperTypes().add(this.getArchitecturalElement());
		interfaceTypeEClass.getESuperTypes().add(this.getArchitecturalElement());
		parameterTypeEClass.getESuperTypes().add(this.getArchitecturalElement());
		componentEClass.getESuperTypes().add(this.getArchitecturalElement());
		replicaEClass.getESuperTypes().add(this.getArchitecturalElement());
		parameterEClass.getESuperTypes().add(this.getArchitecturalElement());
		methodSpecificationEClass.getESuperTypes().add(this.getArchitecturalElement());
		interfaceEClass.getESuperTypes().add(this.getArchitecturalElement());
		requiredInterfaceEClass.getESuperTypes().add(this.getInterface());
		providedInterfaceEClass.getESuperTypes().add(this.getInterface());
		connectorEClass.getESuperTypes().add(this.getArchitecturalElement());
		failureEClass.getESuperTypes().add(this.getArchitecturalElement());
		performanceStatsEClass.getESuperTypes().add(this.getArchitecturalElement());
		monitoredPropertyEClass.getESuperTypes().add(this.getArchitecturalElement());
		tenantEClass.getESuperTypes().add(this.getArchitecturalElement());
		cfEClass.getESuperTypes().add(this.getIssue());
		cf1EClass.getESuperTypes().add(this.getCF());
		cf2EClass.getESuperTypes().add(this.getCF());
		cf3EClass.getESuperTypes().add(this.getCF());
		cf5EClass.getESuperTypes().add(this.getCF());
		moveReplicaEClass.getESuperTypes().add(this.getRule());
		addReplicaEClass.getESuperTypes().add(this.getRule());
		removeReplicaEClass.getESuperTypes().add(this.getRule());
		hwRedeployComponentEClass.getESuperTypes().add(this.getRule());
		lwRedeployComponentEClass.getESuperTypes().add(this.getRule());
		restartComponentEClass.getESuperTypes().add(this.getRule());
		replaceComponentEClass.getESuperTypes().add(this.getRule());
		cf4EClass.getESuperTypes().add(this.getCF());
		recreateConnectorEClass.getESuperTypes().add(this.getRule());

		// Initialize classes and features; add operations and parameters
		initEClass(cloudEClass, Cloud.class, "Cloud", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getCloud_Servers(), this.getServer(), this.getServer_Cloud(), "servers", null, 1, -1, Cloud.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getCloud_Architecture(), this.getArchitecture(), this.getArchitecture_Cloud(), "architecture", null, 1, 1, Cloud.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getCloud_Name(), ecorePackage.getEString(), "name", null, 0, 1, Cloud.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(serverEClass, Server.class, "Server", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getServer_Cloud(), this.getCloud(), this.getCloud_Servers(), "cloud", null, 1, 1, Server.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getServer_Name(), ecorePackage.getEString(), "name", null, 0, 1, Server.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getServer_DeployedReplicas(), this.getReplica(), this.getReplica_DeployedOn(), "deployedReplicas", null, 0, -1, Server.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getServer_DeployedComponent(), this.getComponent(), this.getComponent_DeployedOn(), "deployedComponent", null, 1, -1, Server.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getServer_CPUCapacity(), ecorePackage.getEDouble(), "CPUCapacity", null, 0, 1, Server.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getServer_MemoryCapacity(), ecorePackage.getEDouble(), "MemoryCapacity", null, 0, 1, Server.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getServer_StorageCapacity(), ecorePackage.getEDouble(), "StorageCapacity", null, 0, 1, Server.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getServer_CPUConsumption(), ecorePackage.getEDouble(), "CPUConsumption", null, 0, 1, Server.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getServer_MemoryConsumption(), ecorePackage.getEDouble(), "MemoryConsumption", null, 0, 1, Server.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getServer_StorageConsumption(), ecorePackage.getEDouble(), "StorageConsumption", null, 0, 1, Server.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getServer_Cost(), ecorePackage.getEDouble(), "Cost", null, 0, 1, Server.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getServer_CpuFactor(), ecorePackage.getEDouble(), "cpuFactor", null, 0, 1, Server.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getServer_MemoryFactor(), ecorePackage.getEDouble(), "memoryFactor", null, 0, 1, Server.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getServer_StorageFactor(), ecorePackage.getEDouble(), "storageFactor", null, 0, 1, Server.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(architectureEClass, Architecture.class, "Architecture", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getArchitecture_Name(), ecorePackage.getEString(), "name", null, 0, 1, Architecture.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getArchitecture_ComponentTypes(), this.getComponentType(), this.getComponentType_Architecture(), "componentTypes", null, 0, -1, Architecture.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getArchitecture_InterfaceTypes(), this.getInterfaceType(), this.getInterfaceType_Architecture(), "interfaceTypes", null, 0, -1, Architecture.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getArchitecture_Tenants(), this.getTenant(), this.getTenant_Architecture(), "tenants", null, 0, -1, Architecture.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getArchitecture_Cloud(), this.getCloud(), this.getCloud_Architecture(), "cloud", null, 1, 1, Architecture.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getArchitecture_Annotations(), this.getAnnotations(), this.getAnnotations_Architecture(), "annotations", null, 0, 1, Architecture.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(architecturalElementEClass, ArchitecturalElement.class, "ArchitecturalElement", IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getArchitecturalElement_Uid(), ecorePackage.getEString(), "uid", null, 1, 1, ArchitecturalElement.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getArchitecturalElement_Criticality(), ecorePackage.getEDouble(), "criticality", "0.0", 0, 1, ArchitecturalElement.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getArchitecturalElement_Importance(), ecorePackage.getEDouble(), "importance", null, 0, 1, ArchitecturalElement.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getArchitecturalElement_ADT(), ecorePackage.getEDouble(), "aDT", null, 0, 1, ArchitecturalElement.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getArchitecturalElement_Efficiency(), ecorePackage.getEDouble(), "efficiency", "0.0", 0, 1, ArchitecturalElement.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getArchitecturalElement_Performance(), ecorePackage.getEDouble(), "performance", "0.0", 0, 1, ArchitecturalElement.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getArchitecturalElement_Request(), ecorePackage.getEDouble(), "request", null, 0, 1, ArchitecturalElement.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(componentTypeEClass, ComponentType.class, "ComponentType", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getComponentType_Name(), ecorePackage.getEString(), "name", null, 0, 1, ComponentType.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getComponentType_Architecture(), this.getArchitecture(), this.getArchitecture_ComponentTypes(), "architecture", null, 1, 1, ComponentType.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getComponentType_RequiredInterfaceTypes(), this.getInterfaceType(), this.getInterfaceType_RequiringComponentTypes(), "requiredInterfaceTypes", null, 0, -1, ComponentType.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getComponentType_ProvidedInterfaceTypes(), this.getInterfaceType(), this.getInterfaceType_ProvidingComponentTypes(), "providedInterfaceTypes", null, 1, -1, ComponentType.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getComponentType_ParameterTypes(), this.getParameterType(), this.getParameterType_ComponentType(), "parameterTypes", null, 0, -1, ComponentType.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getComponentType_Instances(), this.getComponent(), this.getComponent_Type(), "instances", null, 0, -1, ComponentType.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getComponentType_Reliability(), ecorePackage.getEDouble(), "reliability", "1", 0, 1, ComponentType.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getComponentType_SatPoint(), ecorePackage.getEDouble(), "satPoint", null, 0, 1, ComponentType.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getComponentType_PerformanceMax(), ecorePackage.getEDouble(), "performanceMax", null, 0, 1, ComponentType.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getComponentType_ReplicaCost(), ecorePackage.getEDouble(), "replicaCost", null, 0, 1, ComponentType.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getComponentType_ReplicaADT(), ecorePackage.getEDouble(), "replicaADT", null, 0, 1, ComponentType.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getComponentType_CpuFactor(), ecorePackage.getEDouble(), "cpuFactor", null, 0, 1, ComponentType.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getComponentType_MemoryFactor(), ecorePackage.getEDouble(), "memoryFactor", null, 0, 1, ComponentType.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getComponentType_StorageFactor(), ecorePackage.getEDouble(), "storageFactor", null, 0, 1, ComponentType.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		addEOperation(componentTypeEClass, this.getComponent(), "instantiate", 0, 1, IS_UNIQUE, IS_ORDERED);

		initEClass(interfaceTypeEClass, InterfaceType.class, "InterfaceType", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getInterfaceType_FullyQualifiedName(), ecorePackage.getEString(), "fullyQualifiedName", null, 0, 1, InterfaceType.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getInterfaceType_Architecture(), this.getArchitecture(), this.getArchitecture_InterfaceTypes(), "architecture", null, 1, 1, InterfaceType.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getInterfaceType_RequiringComponentTypes(), this.getComponentType(), this.getComponentType_RequiredInterfaceTypes(), "requiringComponentTypes", null, 0, -1, InterfaceType.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getInterfaceType_ProvidingComponentTypes(), this.getComponentType(), this.getComponentType_ProvidedInterfaceTypes(), "providingComponentTypes", null, 0, -1, InterfaceType.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getInterfaceType_MethodSpecification(), this.getMethodSpecification(), this.getMethodSpecification_InterfaceType(), "methodSpecification", null, 1, -1, InterfaceType.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getInterfaceType_Instances(), this.getInterface(), this.getInterface_Type(), "instances", null, 0, -1, InterfaceType.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(parameterTypeEClass, ParameterType.class, "ParameterType", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getParameterType_Name(), ecorePackage.getEString(), "name", null, 1, 1, ParameterType.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getParameterType_Type(), ecorePackage.getEString(), "type", null, 1, 1, ParameterType.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getParameterType_DefaultValue(), ecorePackage.getEString(), "defaultValue", null, 0, 1, ParameterType.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getParameterType_ComponentType(), this.getComponentType(), this.getComponentType_ParameterTypes(), "componentType", null, 1, 1, ParameterType.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getParameterType_Instances(), this.getParameter(), this.getParameter_Type(), "instances", null, 0, -1, ParameterType.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(componentEClass, Component.class, "Component", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getComponent_Parameters(), this.getParameter(), this.getParameter_Component(), "parameters", null, 0, -1, Component.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getComponent_RequiredInterfaces(), this.getRequiredInterface(), this.getRequiredInterface_Component(), "requiredInterfaces", null, 0, -1, Component.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getComponent_State(), this.getComponentLifeCycle(), "state", "DEPLOYED", 1, 1, Component.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getComponent_ProvidedInterfaces(), this.getProvidedInterface(), this.getProvidedInterface_Component(), "providedInterfaces", null, 1, -1, Component.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getComponent_Type(), this.getComponentType(), this.getComponentType_Instances(), "type", null, 1, 1, Component.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getComponent_MonitoredProperties(), this.getMonitoredProperty(), this.getMonitoredProperty_Component(), "monitoredProperties", null, 0, -1, Component.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getComponent_Tenant(), this.getTenant(), this.getTenant_Components(), "tenant", null, 1, 1, Component.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getComponent_Issues(), this.getIssue(), this.getIssue_AffectedComponent(), "issues", null, 0, -1, Component.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getComponent_InUseReplica(), ecorePackage.getEDouble(), "inUseReplica", null, 0, 1, Component.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getComponent_HealthStatus(), ecorePackage.getEDouble(), "healthStatus", null, 0, 1, Component.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getComponent_PerformanceOld(), ecorePackage.getEDouble(), "performanceOld", null, 0, 1, Component.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getComponent_OriginalRequest(), ecorePackage.getEDouble(), "originalRequest", null, 0, 1, Component.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getComponent_DeployedOn(), this.getServer(), this.getServer_DeployedComponent(), "deployedOn", null, 1, 1, Component.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getComponent_Replica(), this.getReplica(), this.getReplica_Component(), "replica", null, 0, 1, Component.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(replicaEClass, Replica.class, "Replica", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getReplica_DeployedOn(), this.getServer(), this.getServer_DeployedReplicas(), "deployedOn", null, 1, 1, Replica.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getReplica_Cost(), ecorePackage.getEDouble(), "cost", null, 0, 1, Replica.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getReplica_InUse(), ecorePackage.getEDouble(), "inUse", null, 0, 1, Replica.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getReplica_CPURequirement(), ecorePackage.getEDouble(), "CPURequirement", null, 0, 1, Replica.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getReplica_MemoryRequirement(), ecorePackage.getEDouble(), "MemoryRequirement", null, 0, 1, Replica.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getReplica_StorageRequirement(), ecorePackage.getEDouble(), "StorageRequirement", null, 0, 1, Replica.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getReplica_Component(), this.getComponent(), this.getComponent_Replica(), "component", null, 1, 1, Replica.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(parameterEClass, Parameter.class, "Parameter", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getParameter_Value(), ecorePackage.getEString(), "value", null, 1, 1, Parameter.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getParameter_LazyAdaptation(), ecorePackage.getEBoolean(), "lazyAdaptation", null, 1, 1, Parameter.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getParameter_Component(), this.getComponent(), this.getComponent_Parameters(), "component", null, 1, 1, Parameter.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getParameter_Type(), this.getParameterType(), this.getParameterType_Instances(), "type", null, 1, 1, Parameter.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(methodSpecificationEClass, MethodSpecification.class, "MethodSpecification", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getMethodSpecification_Signature(), ecorePackage.getEString(), "signature", null, 1, 1, MethodSpecification.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getMethodSpecification_InterfaceType(), this.getInterfaceType(), this.getInterfaceType_MethodSpecification(), "interfaceType", null, 1, 1, MethodSpecification.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(interfaceEClass, Interface.class, "Interface", IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getInterface_Type(), this.getInterfaceType(), this.getInterfaceType_Instances(), "type", null, 1, 1, Interface.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(requiredInterfaceEClass, RequiredInterface.class, "RequiredInterface", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getRequiredInterface_Component(), this.getComponent(), this.getComponent_RequiredInterfaces(), "component", null, 1, 1, RequiredInterface.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getRequiredInterface_Connector(), this.getConnector(), this.getConnector_Source(), "connector", null, 0, 1, RequiredInterface.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(providedInterfaceEClass, ProvidedInterface.class, "ProvidedInterface", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getProvidedInterface_Component(), this.getComponent(), this.getComponent_ProvidedInterfaces(), "component", null, 1, 1, ProvidedInterface.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getProvidedInterface_Connectors(), this.getConnector(), this.getConnector_Target(), "connectors", null, 0, -1, ProvidedInterface.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getProvidedInterface_Failures(), this.getFailure(), this.getFailure_Interface(), "failures", null, 0, -1, ProvidedInterface.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getProvidedInterface_PerformanceStats(), this.getPerformanceStats(), this.getPerformanceStats_Interface(), "performanceStats", null, 0, -1, ProvidedInterface.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(connectorEClass, Connector.class, "Connector", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getConnector_LazyAdaptation(), ecorePackage.getEBoolean(), "lazyAdaptation", null, 1, 1, Connector.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getConnector_Source(), this.getRequiredInterface(), this.getRequiredInterface_Connector(), "source", null, 1, 1, Connector.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getConnector_Target(), this.getProvidedInterface(), this.getProvidedInterface_Connectors(), "target", null, 1, 1, Connector.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(failureEClass, Failure.class, "Failure", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getFailure_Name(), ecorePackage.getEString(), "name", null, 1, 1, Failure.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getFailure_Message(), ecorePackage.getEString(), "message", null, 0, 1, Failure.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getFailure_Interface(), this.getProvidedInterface(), this.getProvidedInterface_Failures(), "interface", null, 1, 1, Failure.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getFailure_Method(), this.getMethodSpecification(), null, "method", null, 1, 1, Failure.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(performanceStatsEClass, PerformanceStats.class, "PerformanceStats", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getPerformanceStats_MinTime(), ecorePackage.getELong(), "minTime", null, 0, 1, PerformanceStats.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getPerformanceStats_MaxTime(), ecorePackage.getELong(), "maxTime", null, 0, 1, PerformanceStats.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getPerformanceStats_TotalTime(), ecorePackage.getELong(), "totalTime", null, 0, 1, PerformanceStats.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getPerformanceStats_InvocationCount(), ecorePackage.getELong(), "invocationCount", null, 0, 1, PerformanceStats.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getPerformanceStats_Interface(), this.getProvidedInterface(), this.getProvidedInterface_PerformanceStats(), "interface", null, 1, 1, PerformanceStats.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getPerformanceStats_Method(), this.getMethodSpecification(), null, "method", null, 1, 1, PerformanceStats.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(monitoredPropertyEClass, MonitoredProperty.class, "MonitoredProperty", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getMonitoredProperty_Name(), ecorePackage.getEString(), "name", null, 1, 1, MonitoredProperty.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getMonitoredProperty_Value(), ecorePackage.getEString(), "value", null, 1, 1, MonitoredProperty.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getMonitoredProperty_Type(), ecorePackage.getEString(), "type", null, 1, 1, MonitoredProperty.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getMonitoredProperty_Unit(), ecorePackage.getEString(), "unit", null, 1, 1, MonitoredProperty.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getMonitoredProperty_Description(), ecorePackage.getEString(), "description", null, 0, 1, MonitoredProperty.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getMonitoredProperty_Component(), this.getComponent(), this.getComponent_MonitoredProperties(), "component", null, 1, 1, MonitoredProperty.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getMonitoredProperty_ConcernedElements(), this.getArchitecturalElement(), null, "concernedElements", null, 0, -1, MonitoredProperty.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(tenantEClass, Tenant.class, "Tenant", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getTenant_Name(), ecorePackage.getEString(), "name", "", 0, 1, Tenant.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getTenant_Architecture(), this.getArchitecture(), this.getArchitecture_Tenants(), "architecture", null, 1, 1, Tenant.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getTenant_Components(), this.getComponent(), this.getComponent_Tenant(), "components", null, 0, -1, Tenant.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getTenant_ReplicaFactor(), ecorePackage.getEDouble(), "replicaFactor", null, 0, 1, Tenant.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(annotationsEClass, Annotations.class, "Annotations", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getAnnotations_Architecture(), this.getArchitecture(), this.getArchitecture_Annotations(), "architecture", null, 1, 1, Annotations.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getAnnotations_Issues(), this.getIssue(), this.getIssue_Annotations(), "issues", null, 0, -1, Annotations.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getAnnotations_Rules(), this.getRule(), this.getRule_Annotations(), "rules", null, 0, -1, Annotations.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(issueEClass, Issue.class, "Issue", IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getIssue_UtilityDrop(), ecorePackage.getEDouble(), "utilityDrop", null, 0, 1, Issue.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getIssue_AffectedComponent(), this.getComponent(), this.getComponent_Issues(), "affectedComponent", null, 1, 1, Issue.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getIssue_Annotations(), this.getAnnotations(), this.getAnnotations_Issues(), "annotations", null, 1, 1, Issue.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getIssue_HandledBy(), this.getRule(), this.getRule_Handles(), "handledBy", null, 0, -1, Issue.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(cfEClass, de.mdelab.morisia.comparch.CF.class, "CF", IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);

		initEClass(cf1EClass, de.mdelab.morisia.comparch.CF1.class, "CF1", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);

		initEClass(cf2EClass, de.mdelab.morisia.comparch.CF2.class, "CF2", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);

		initEClass(cf3EClass, de.mdelab.morisia.comparch.CF3.class, "CF3", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);

		initEClass(cf5EClass, de.mdelab.morisia.comparch.CF5.class, "CF5", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);

		initEClass(ruleEClass, Rule.class, "Rule", IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEAttribute(getRule_Costs(), ecorePackage.getEDouble(), "costs", null, 0, 1, Rule.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getRule_Annotations(), this.getAnnotations(), this.getAnnotations_Rules(), "annotations", null, 1, 1, Rule.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, !IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEReference(getRule_Handles(), this.getIssue(), this.getIssue_HandledBy(), "handles", null, 1, 1, Rule.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);
		initEAttribute(getRule_UtilityIncrease(), ecorePackage.getEDouble(), "utilityIncrease", null, 0, 1, Rule.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_UNSETTABLE, !IS_ID, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(moveReplicaEClass, MoveReplica.class, "MoveReplica", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);

		initEClass(addReplicaEClass, AddReplica.class, "AddReplica", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);

		initEClass(removeReplicaEClass, RemoveReplica.class, "RemoveReplica", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);

		initEClass(hwRedeployComponentEClass, HwRedeployComponent.class, "HwRedeployComponent", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);

		initEClass(lwRedeployComponentEClass, LwRedeployComponent.class, "LwRedeployComponent", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);

		initEClass(restartComponentEClass, RestartComponent.class, "RestartComponent", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);

		initEClass(replaceComponentEClass, ReplaceComponent.class, "ReplaceComponent", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);
		initEReference(getReplaceComponent_AlternativeComponentType(), this.getComponentType(), null, "alternativeComponentType", null, 0, 1, ReplaceComponent.class, !IS_TRANSIENT, !IS_VOLATILE, IS_CHANGEABLE, !IS_COMPOSITE, IS_RESOLVE_PROXIES, !IS_UNSETTABLE, IS_UNIQUE, !IS_DERIVED, IS_ORDERED);

		initEClass(cf4EClass, de.mdelab.morisia.comparch.CF4.class, "CF4", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);

		initEClass(recreateConnectorEClass, RecreateConnector.class, "RecreateConnector", !IS_ABSTRACT, !IS_INTERFACE, IS_GENERATED_INSTANCE_CLASS);

		// Initialize enums and add enum literals
		initEEnum(componentLifeCycleEEnum, ComponentLifeCycle.class, "ComponentLifeCycle");
		addEEnumLiteral(componentLifeCycleEEnum, ComponentLifeCycle.DEPLOYED);
		addEEnumLiteral(componentLifeCycleEEnum, ComponentLifeCycle.STARTED);
		addEEnumLiteral(componentLifeCycleEEnum, ComponentLifeCycle.UNDEPLOYED);
		addEEnumLiteral(componentLifeCycleEEnum, ComponentLifeCycle.NOT_SUPPORTED);
		addEEnumLiteral(componentLifeCycleEEnum, ComponentLifeCycle.REMOVED);

		// Create resource
		createResource(eNS_URI);

		// Create annotations
		// http://www.obeo.fr/dsl/dnc/archetype
		createArchetypeAnnotations();
	}

	/**
	 * Initializes the annotations for <b>http://www.obeo.fr/dsl/dnc/archetype</b>.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected void createArchetypeAnnotations() {
		String source = "http://www.obeo.fr/dsl/dnc/archetype";	
		addAnnotation
		  (addReplicaEClass, 
		   source, 
		   new String[] {
			 "archetype", "Thing"
		   });	
		addAnnotation
		  (removeReplicaEClass, 
		   source, 
		   new String[] {
			 "archetype", "Thing"
		   });
	}

} //ComparchPackageImpl
