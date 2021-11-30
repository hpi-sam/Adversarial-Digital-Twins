/**
 */
package de.mdelab.morisia.comparch;


/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Replica</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * </p>
 * <ul>
 *   <li>{@link de.mdelab.morisia.comparch.Replica#getDeployedOn <em>Deployed On</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Replica#getCost <em>Cost</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Replica#getInUse <em>In Use</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Replica#getCPURequirement <em>CPU Requirement</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Replica#getMemoryRequirement <em>Memory Requirement</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Replica#getStorageRequirement <em>Storage Requirement</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Replica#getComponent <em>Component</em>}</li>
 * </ul>
 *
 * @see de.mdelab.morisia.comparch.ComparchPackage#getReplica()
 * @model
 * @generated
 */
public interface Replica extends ArchitecturalElement {
	/**
	 * Returns the value of the '<em><b>Deployed On</b></em>' reference.
	 * It is bidirectional and its opposite is '{@link de.mdelab.morisia.comparch.Server#getDeployedReplicas <em>Deployed Replicas</em>}'.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Deployed On</em>' reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Deployed On</em>' reference.
	 * @see #setDeployedOn(Server)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getReplica_DeployedOn()
	 * @see de.mdelab.morisia.comparch.Server#getDeployedReplicas
	 * @model opposite="deployedReplicas" required="true"
	 * @generated
	 */
	Server getDeployedOn();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Replica#getDeployedOn <em>Deployed On</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Deployed On</em>' reference.
	 * @see #getDeployedOn()
	 * @generated
	 */
	void setDeployedOn(Server value);

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
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getReplica_Cost()
	 * @model
	 * @generated
	 */
	double getCost();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Replica#getCost <em>Cost</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Cost</em>' attribute.
	 * @see #getCost()
	 * @generated
	 */
	void setCost(double value);

	/**
	 * Returns the value of the '<em><b>In Use</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>In Use</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>In Use</em>' attribute.
	 * @see #setInUse(double)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getReplica_InUse()
	 * @model
	 * @generated
	 */
	double getInUse();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Replica#getInUse <em>In Use</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>In Use</em>' attribute.
	 * @see #getInUse()
	 * @generated
	 */
	void setInUse(double value);

	/**
	 * Returns the value of the '<em><b>CPU Requirement</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>CPU Requirement</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>CPU Requirement</em>' attribute.
	 * @see #setCPURequirement(double)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getReplica_CPURequirement()
	 * @model
	 * @generated
	 */
	double getCPURequirement();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Replica#getCPURequirement <em>CPU Requirement</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>CPU Requirement</em>' attribute.
	 * @see #getCPURequirement()
	 * @generated
	 */
	void setCPURequirement(double value);

	/**
	 * Returns the value of the '<em><b>Memory Requirement</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Memory Requirement</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Memory Requirement</em>' attribute.
	 * @see #setMemoryRequirement(double)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getReplica_MemoryRequirement()
	 * @model
	 * @generated
	 */
	double getMemoryRequirement();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Replica#getMemoryRequirement <em>Memory Requirement</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Memory Requirement</em>' attribute.
	 * @see #getMemoryRequirement()
	 * @generated
	 */
	void setMemoryRequirement(double value);

	/**
	 * Returns the value of the '<em><b>Storage Requirement</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Storage Requirement</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Storage Requirement</em>' attribute.
	 * @see #setStorageRequirement(double)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getReplica_StorageRequirement()
	 * @model
	 * @generated
	 */
	double getStorageRequirement();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Replica#getStorageRequirement <em>Storage Requirement</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Storage Requirement</em>' attribute.
	 * @see #getStorageRequirement()
	 * @generated
	 */
	void setStorageRequirement(double value);

	/**
	 * Returns the value of the '<em><b>Component</b></em>' container reference.
	 * It is bidirectional and its opposite is '{@link de.mdelab.morisia.comparch.Component#getReplica <em>Replica</em>}'.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Component</em>' container reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Component</em>' container reference.
	 * @see #setComponent(Component)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getReplica_Component()
	 * @see de.mdelab.morisia.comparch.Component#getReplica
	 * @model opposite="replica" required="true" transient="false"
	 * @generated
	 */
	Component getComponent();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Replica#getComponent <em>Component</em>}' container reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Component</em>' container reference.
	 * @see #getComponent()
	 * @generated
	 */
	void setComponent(Component value);

} // Replica
