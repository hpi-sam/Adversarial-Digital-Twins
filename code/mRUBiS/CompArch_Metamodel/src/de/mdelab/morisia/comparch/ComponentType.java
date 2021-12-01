/**
 */
package de.mdelab.morisia.comparch;

import org.eclipse.emf.common.util.EList;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Component Type</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * </p>
 * <ul>
 *   <li>{@link de.mdelab.morisia.comparch.ComponentType#getName <em>Name</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.ComponentType#getArchitecture <em>Architecture</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.ComponentType#getRequiredInterfaceTypes <em>Required Interface Types</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.ComponentType#getProvidedInterfaceTypes <em>Provided Interface Types</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.ComponentType#getParameterTypes <em>Parameter Types</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.ComponentType#getInstances <em>Instances</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.ComponentType#getReliability <em>Reliability</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.ComponentType#getSatPoint <em>Sat Point</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.ComponentType#getPerformanceMax <em>Performance Max</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.ComponentType#getReplicaCost <em>Replica Cost</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.ComponentType#getReplicaADT <em>Replica ADT</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.ComponentType#getCpuFactor <em>Cpu Factor</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.ComponentType#getMemoryFactor <em>Memory Factor</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.ComponentType#getStorageFactor <em>Storage Factor</em>}</li>
 * </ul>
 *
 * @see de.mdelab.morisia.comparch.ComparchPackage#getComponentType()
 * @model
 * @generated
 */
public interface ComponentType extends ArchitecturalElement {
	/**
	 * Returns the value of the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Name</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Name</em>' attribute.
	 * @see #setName(String)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getComponentType_Name()
	 * @model
	 * @generated
	 */
	String getName();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.ComponentType#getName <em>Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Name</em>' attribute.
	 * @see #getName()
	 * @generated
	 */
	void setName(String value);

	/**
	 * Returns the value of the '<em><b>Architecture</b></em>' container reference.
	 * It is bidirectional and its opposite is '{@link de.mdelab.morisia.comparch.Architecture#getComponentTypes <em>Component Types</em>}'.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Architecture</em>' container reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Architecture</em>' container reference.
	 * @see #setArchitecture(Architecture)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getComponentType_Architecture()
	 * @see de.mdelab.morisia.comparch.Architecture#getComponentTypes
	 * @model opposite="componentTypes" required="true" transient="false"
	 * @generated
	 */
	Architecture getArchitecture();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.ComponentType#getArchitecture <em>Architecture</em>}' container reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Architecture</em>' container reference.
	 * @see #getArchitecture()
	 * @generated
	 */
	void setArchitecture(Architecture value);

	/**
	 * Returns the value of the '<em><b>Required Interface Types</b></em>' reference list.
	 * The list contents are of type {@link de.mdelab.morisia.comparch.InterfaceType}.
	 * It is bidirectional and its opposite is '{@link de.mdelab.morisia.comparch.InterfaceType#getRequiringComponentTypes <em>Requiring Component Types</em>}'.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Required Interface Types</em>' reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Required Interface Types</em>' reference list.
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getComponentType_RequiredInterfaceTypes()
	 * @see de.mdelab.morisia.comparch.InterfaceType#getRequiringComponentTypes
	 * @model opposite="requiringComponentTypes"
	 * @generated
	 */
	EList<InterfaceType> getRequiredInterfaceTypes();

	/**
	 * Returns the value of the '<em><b>Provided Interface Types</b></em>' reference list.
	 * The list contents are of type {@link de.mdelab.morisia.comparch.InterfaceType}.
	 * It is bidirectional and its opposite is '{@link de.mdelab.morisia.comparch.InterfaceType#getProvidingComponentTypes <em>Providing Component Types</em>}'.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Provided Interface Types</em>' reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Provided Interface Types</em>' reference list.
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getComponentType_ProvidedInterfaceTypes()
	 * @see de.mdelab.morisia.comparch.InterfaceType#getProvidingComponentTypes
	 * @model opposite="providingComponentTypes" required="true"
	 * @generated
	 */
	EList<InterfaceType> getProvidedInterfaceTypes();

	/**
	 * Returns the value of the '<em><b>Parameter Types</b></em>' containment reference list.
	 * The list contents are of type {@link de.mdelab.morisia.comparch.ParameterType}.
	 * It is bidirectional and its opposite is '{@link de.mdelab.morisia.comparch.ParameterType#getComponentType <em>Component Type</em>}'.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Parameter Types</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Parameter Types</em>' containment reference list.
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getComponentType_ParameterTypes()
	 * @see de.mdelab.morisia.comparch.ParameterType#getComponentType
	 * @model opposite="componentType" containment="true"
	 * @generated
	 */
	EList<ParameterType> getParameterTypes();

	/**
	 * Returns the value of the '<em><b>Instances</b></em>' reference list.
	 * The list contents are of type {@link de.mdelab.morisia.comparch.Component}.
	 * It is bidirectional and its opposite is '{@link de.mdelab.morisia.comparch.Component#getType <em>Type</em>}'.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Instances</em>' reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Instances</em>' reference list.
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getComponentType_Instances()
	 * @see de.mdelab.morisia.comparch.Component#getType
	 * @model opposite="type"
	 * @generated
	 */
	EList<Component> getInstances();

	/**
	 * Returns the value of the '<em><b>Reliability</b></em>' attribute.
	 * The default value is <code>"1"</code>.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Reliability</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Reliability</em>' attribute.
	 * @see #setReliability(double)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getComponentType_Reliability()
	 * @model default="1"
	 * @generated
	 */
	double getReliability();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.ComponentType#getReliability <em>Reliability</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Reliability</em>' attribute.
	 * @see #getReliability()
	 * @generated
	 */
	void setReliability(double value);

	/**
	 * Returns the value of the '<em><b>Sat Point</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Sat Point</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Sat Point</em>' attribute.
	 * @see #setSatPoint(double)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getComponentType_SatPoint()
	 * @model
	 * @generated
	 */
	double getSatPoint();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.ComponentType#getSatPoint <em>Sat Point</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Sat Point</em>' attribute.
	 * @see #getSatPoint()
	 * @generated
	 */
	void setSatPoint(double value);

	/**
	 * Returns the value of the '<em><b>Performance Max</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Performance Max</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Performance Max</em>' attribute.
	 * @see #setPerformanceMax(double)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getComponentType_PerformanceMax()
	 * @model
	 * @generated
	 */
	double getPerformanceMax();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.ComponentType#getPerformanceMax <em>Performance Max</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Performance Max</em>' attribute.
	 * @see #getPerformanceMax()
	 * @generated
	 */
	void setPerformanceMax(double value);

	/**
	 * Returns the value of the '<em><b>Replica Cost</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Replica Cost</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Replica Cost</em>' attribute.
	 * @see #setReplicaCost(double)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getComponentType_ReplicaCost()
	 * @model
	 * @generated
	 */
	double getReplicaCost();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.ComponentType#getReplicaCost <em>Replica Cost</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Replica Cost</em>' attribute.
	 * @see #getReplicaCost()
	 * @generated
	 */
	void setReplicaCost(double value);

	/**
	 * Returns the value of the '<em><b>Replica ADT</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Replica ADT</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Replica ADT</em>' attribute.
	 * @see #setReplicaADT(double)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getComponentType_ReplicaADT()
	 * @model
	 * @generated
	 */
	double getReplicaADT();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.ComponentType#getReplicaADT <em>Replica ADT</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Replica ADT</em>' attribute.
	 * @see #getReplicaADT()
	 * @generated
	 */
	void setReplicaADT(double value);

	/**
	 * Returns the value of the '<em><b>Cpu Factor</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Cpu Factor</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Cpu Factor</em>' attribute.
	 * @see #setCpuFactor(double)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getComponentType_CpuFactor()
	 * @model
	 * @generated
	 */
	double getCpuFactor();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.ComponentType#getCpuFactor <em>Cpu Factor</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Cpu Factor</em>' attribute.
	 * @see #getCpuFactor()
	 * @generated
	 */
	void setCpuFactor(double value);

	/**
	 * Returns the value of the '<em><b>Memory Factor</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Memory Factor</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Memory Factor</em>' attribute.
	 * @see #setMemoryFactor(double)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getComponentType_MemoryFactor()
	 * @model
	 * @generated
	 */
	double getMemoryFactor();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.ComponentType#getMemoryFactor <em>Memory Factor</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Memory Factor</em>' attribute.
	 * @see #getMemoryFactor()
	 * @generated
	 */
	void setMemoryFactor(double value);

	/**
	 * Returns the value of the '<em><b>Storage Factor</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Storage Factor</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Storage Factor</em>' attribute.
	 * @see #setStorageFactor(double)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getComponentType_StorageFactor()
	 * @model
	 * @generated
	 */
	double getStorageFactor();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.ComponentType#getStorageFactor <em>Storage Factor</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Storage Factor</em>' attribute.
	 * @see #getStorageFactor()
	 * @generated
	 */
	void setStorageFactor(double value);

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @model
	 * @generated
	 */
	Component instantiate();

} // ComponentType
