/**
 */
package de.mdelab.morisia.comparch;

import org.eclipse.emf.common.util.EList;
import org.eclipse.emf.ecore.EObject;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Server</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * </p>
 * <ul>
 *   <li>{@link de.mdelab.morisia.comparch.Server#getCloud <em>Cloud</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Server#getName <em>Name</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Server#getDeployedReplicas <em>Deployed Replicas</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Server#getDeployedComponent <em>Deployed Component</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Server#getCPUCapacity <em>CPU Capacity</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Server#getMemoryCapacity <em>Memory Capacity</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Server#getStorageCapacity <em>Storage Capacity</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Server#getCPUConsumption <em>CPU Consumption</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Server#getMemoryConsumption <em>Memory Consumption</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Server#getStorageConsumption <em>Storage Consumption</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Server#getCost <em>Cost</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Server#getCpuFactor <em>Cpu Factor</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Server#getMemoryFactor <em>Memory Factor</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Server#getStorageFactor <em>Storage Factor</em>}</li>
 * </ul>
 *
 * @see de.mdelab.morisia.comparch.ComparchPackage#getServer()
 * @model
 * @generated
 */
public interface Server extends ArchitecturalElement {

	/**
	 * Returns the value of the '<em><b>CPU Capacity</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>CPU Capacity</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>CPU Capacity</em>' attribute.
	 * @see #setCPUCapacity(double)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getServer_CPUCapacity()
	 * @model
	 * @generated
	 */
	double getCPUCapacity();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Server#getCPUCapacity <em>CPU Capacity</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>CPU Capacity</em>' attribute.
	 * @see #getCPUCapacity()
	 * @generated
	 */
	void setCPUCapacity(double value);

	/**
	 * Returns the value of the '<em><b>Memory Capacity</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Memory Capacity</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Memory Capacity</em>' attribute.
	 * @see #setMemoryCapacity(double)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getServer_MemoryCapacity()
	 * @model
	 * @generated
	 */
	double getMemoryCapacity();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Server#getMemoryCapacity <em>Memory Capacity</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Memory Capacity</em>' attribute.
	 * @see #getMemoryCapacity()
	 * @generated
	 */
	void setMemoryCapacity(double value);

	/**
	 * Returns the value of the '<em><b>Storage Capacity</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Storage Capacity</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Storage Capacity</em>' attribute.
	 * @see #setStorageCapacity(double)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getServer_StorageCapacity()
	 * @model
	 * @generated
	 */
	double getStorageCapacity();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Server#getStorageCapacity <em>Storage Capacity</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Storage Capacity</em>' attribute.
	 * @see #getStorageCapacity()
	 * @generated
	 */
	void setStorageCapacity(double value);

	/**
	 * Returns the value of the '<em><b>CPU Consumption</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>CPU Consumption</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>CPU Consumption</em>' attribute.
	 * @see #setCPUConsumption(double)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getServer_CPUConsumption()
	 * @model
	 * @generated
	 */
	double getCPUConsumption();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Server#getCPUConsumption <em>CPU Consumption</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>CPU Consumption</em>' attribute.
	 * @see #getCPUConsumption()
	 * @generated
	 */
	void setCPUConsumption(double value);

	/**
	 * Returns the value of the '<em><b>Memory Consumption</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Memory Consumption</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Memory Consumption</em>' attribute.
	 * @see #setMemoryConsumption(double)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getServer_MemoryConsumption()
	 * @model
	 * @generated
	 */
	double getMemoryConsumption();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Server#getMemoryConsumption <em>Memory Consumption</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Memory Consumption</em>' attribute.
	 * @see #getMemoryConsumption()
	 * @generated
	 */
	void setMemoryConsumption(double value);

	/**
	 * Returns the value of the '<em><b>Storage Consumption</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Storage Consumption</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Storage Consumption</em>' attribute.
	 * @see #setStorageConsumption(double)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getServer_StorageConsumption()
	 * @model
	 * @generated
	 */
	double getStorageConsumption();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Server#getStorageConsumption <em>Storage Consumption</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Storage Consumption</em>' attribute.
	 * @see #getStorageConsumption()
	 * @generated
	 */
	void setStorageConsumption(double value);

	/**
	 * Returns the value of the '<em><b>Cost</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Cost</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Cost</em>' attribute.
	 * @see #setCost(double)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getServer_Cost()
	 * @model
	 * @generated
	 */
	double getCost();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Server#getCost <em>Cost</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Cost</em>' attribute.
	 * @see #getCost()
	 * @generated
	 */
	void setCost(double value);

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
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getServer_CpuFactor()
	 * @model
	 * @generated
	 */
	double getCpuFactor();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Server#getCpuFactor <em>Cpu Factor</em>}' attribute.
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
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getServer_MemoryFactor()
	 * @model
	 * @generated
	 */
	double getMemoryFactor();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Server#getMemoryFactor <em>Memory Factor</em>}' attribute.
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
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getServer_StorageFactor()
	 * @model
	 * @generated
	 */
	double getStorageFactor();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Server#getStorageFactor <em>Storage Factor</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Storage Factor</em>' attribute.
	 * @see #getStorageFactor()
	 * @generated
	 */
	void setStorageFactor(double value);

	/**
	 * Returns the value of the '<em><b>Cloud</b></em>' container reference.
	 * It is bidirectional and its opposite is '{@link de.mdelab.morisia.comparch.Cloud#getServers <em>Servers</em>}'.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Cloud</em>' container reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Cloud</em>' container reference.
	 * @see #setCloud(Cloud)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getServer_Cloud()
	 * @see de.mdelab.morisia.comparch.Cloud#getServers
	 * @model opposite="servers" required="true" transient="false"
	 * @generated
	 */
	Cloud getCloud();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Server#getCloud <em>Cloud</em>}' container reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Cloud</em>' container reference.
	 * @see #getCloud()
	 * @generated
	 */
	void setCloud(Cloud value);

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
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getServer_Name()
	 * @model
	 * @generated
	 */
	String getName();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Server#getName <em>Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Name</em>' attribute.
	 * @see #getName()
	 * @generated
	 */
	void setName(String value);

	/**
	 * Returns the value of the '<em><b>Deployed Replicas</b></em>' reference list.
	 * The list contents are of type {@link de.mdelab.morisia.comparch.Replica}.
	 * It is bidirectional and its opposite is '{@link de.mdelab.morisia.comparch.Replica#getDeployedOn <em>Deployed On</em>}'.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Deployed Replicas</em>' reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Deployed Replicas</em>' reference list.
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getServer_DeployedReplicas()
	 * @see de.mdelab.morisia.comparch.Replica#getDeployedOn
	 * @model opposite="deployedOn"
	 * @generated
	 */
	EList<Replica> getDeployedReplicas();

	/**
	 * Returns the value of the '<em><b>Deployed Component</b></em>' reference list.
	 * The list contents are of type {@link de.mdelab.morisia.comparch.Component}.
	 * It is bidirectional and its opposite is '{@link de.mdelab.morisia.comparch.Component#getDeployedOn <em>Deployed On</em>}'.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Deployed Component</em>' reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Deployed Component</em>' reference list.
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getServer_DeployedComponent()
	 * @see de.mdelab.morisia.comparch.Component#getDeployedOn
	 * @model opposite="deployedOn" required="true"
	 * @generated
	 */
	EList<Component> getDeployedComponent();
} // Server
