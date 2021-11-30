/**
 */
package de.mdelab.morisia.comparch;

import org.eclipse.emf.common.util.EList;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Component</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * </p>
 * <ul>
 *   <li>{@link de.mdelab.morisia.comparch.Component#getParameters <em>Parameters</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Component#getRequiredInterfaces <em>Required Interfaces</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Component#getState <em>State</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Component#getProvidedInterfaces <em>Provided Interfaces</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Component#getType <em>Type</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Component#getMonitoredProperties <em>Monitored Properties</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Component#getTenant <em>Tenant</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Component#getIssues <em>Issues</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Component#getInUseReplica <em>In Use Replica</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Component#getHealthStatus <em>Health Status</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Component#getPerformanceOld <em>Performance Old</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Component#getOriginalRequest <em>Original Request</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Component#getDeployedOn <em>Deployed On</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Component#getReplica <em>Replica</em>}</li>
 * </ul>
 *
 * @see de.mdelab.morisia.comparch.ComparchPackage#getComponent()
 * @model
 * @generated
 */
public interface Component extends ArchitecturalElement {
	/**
	 * Returns the value of the '<em><b>Parameters</b></em>' containment reference list.
	 * The list contents are of type {@link de.mdelab.morisia.comparch.Parameter}.
	 * It is bidirectional and its opposite is '{@link de.mdelab.morisia.comparch.Parameter#getComponent <em>Component</em>}'.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Parameters</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Parameters</em>' containment reference list.
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getComponent_Parameters()
	 * @see de.mdelab.morisia.comparch.Parameter#getComponent
	 * @model opposite="component" containment="true"
	 * @generated
	 */
	EList<Parameter> getParameters();

	/**
	 * Returns the value of the '<em><b>Required Interfaces</b></em>' containment reference list.
	 * The list contents are of type {@link de.mdelab.morisia.comparch.RequiredInterface}.
	 * It is bidirectional and its opposite is '{@link de.mdelab.morisia.comparch.RequiredInterface#getComponent <em>Component</em>}'.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Required Interfaces</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Required Interfaces</em>' containment reference list.
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getComponent_RequiredInterfaces()
	 * @see de.mdelab.morisia.comparch.RequiredInterface#getComponent
	 * @model opposite="component" containment="true"
	 * @generated
	 */
	EList<RequiredInterface> getRequiredInterfaces();

	/**
	 * Returns the value of the '<em><b>State</b></em>' attribute.
	 * The default value is <code>"DEPLOYED"</code>.
	 * The literals are from the enumeration {@link de.mdelab.morisia.comparch.ComponentLifeCycle}.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>State</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>State</em>' attribute.
	 * @see de.mdelab.morisia.comparch.ComponentLifeCycle
	 * @see #setState(ComponentLifeCycle)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getComponent_State()
	 * @model default="DEPLOYED" required="true"
	 * @generated
	 */
	ComponentLifeCycle getState();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Component#getState <em>State</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>State</em>' attribute.
	 * @see de.mdelab.morisia.comparch.ComponentLifeCycle
	 * @see #getState()
	 * @generated
	 */
	void setState(ComponentLifeCycle value);

	/**
	 * Returns the value of the '<em><b>Provided Interfaces</b></em>' containment reference list.
	 * The list contents are of type {@link de.mdelab.morisia.comparch.ProvidedInterface}.
	 * It is bidirectional and its opposite is '{@link de.mdelab.morisia.comparch.ProvidedInterface#getComponent <em>Component</em>}'.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Provided Interfaces</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Provided Interfaces</em>' containment reference list.
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getComponent_ProvidedInterfaces()
	 * @see de.mdelab.morisia.comparch.ProvidedInterface#getComponent
	 * @model opposite="component" containment="true" required="true"
	 * @generated
	 */
	EList<ProvidedInterface> getProvidedInterfaces();

	/**
	 * Returns the value of the '<em><b>Type</b></em>' reference.
	 * It is bidirectional and its opposite is '{@link de.mdelab.morisia.comparch.ComponentType#getInstances <em>Instances</em>}'.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Type</em>' reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Type</em>' reference.
	 * @see #setType(ComponentType)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getComponent_Type()
	 * @see de.mdelab.morisia.comparch.ComponentType#getInstances
	 * @model opposite="instances" required="true"
	 * @generated
	 */
	ComponentType getType();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Component#getType <em>Type</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Type</em>' reference.
	 * @see #getType()
	 * @generated
	 */
	void setType(ComponentType value);

	/**
	 * Returns the value of the '<em><b>Monitored Properties</b></em>' containment reference list.
	 * The list contents are of type {@link de.mdelab.morisia.comparch.MonitoredProperty}.
	 * It is bidirectional and its opposite is '{@link de.mdelab.morisia.comparch.MonitoredProperty#getComponent <em>Component</em>}'.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Monitored Properties</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Monitored Properties</em>' containment reference list.
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getComponent_MonitoredProperties()
	 * @see de.mdelab.morisia.comparch.MonitoredProperty#getComponent
	 * @model opposite="component" containment="true"
	 * @generated
	 */
	EList<MonitoredProperty> getMonitoredProperties();

	/**
	 * Returns the value of the '<em><b>Tenant</b></em>' container reference.
	 * It is bidirectional and its opposite is '{@link de.mdelab.morisia.comparch.Tenant#getComponents <em>Components</em>}'.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Tenant</em>' container reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Tenant</em>' container reference.
	 * @see #setTenant(Tenant)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getComponent_Tenant()
	 * @see de.mdelab.morisia.comparch.Tenant#getComponents
	 * @model opposite="components" required="true" transient="false"
	 * @generated
	 */
	Tenant getTenant();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Component#getTenant <em>Tenant</em>}' container reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Tenant</em>' container reference.
	 * @see #getTenant()
	 * @generated
	 */
	void setTenant(Tenant value);

	/**
	 * Returns the value of the '<em><b>Issues</b></em>' reference list.
	 * The list contents are of type {@link de.mdelab.morisia.comparch.Issue}.
	 * It is bidirectional and its opposite is '{@link de.mdelab.morisia.comparch.Issue#getAffectedComponent <em>Affected Component</em>}'.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Issues</em>' reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Issues</em>' reference list.
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getComponent_Issues()
	 * @see de.mdelab.morisia.comparch.Issue#getAffectedComponent
	 * @model opposite="affectedComponent"
	 * @generated
	 */
	EList<Issue> getIssues();

	/**
	 * Returns the value of the '<em><b>In Use Replica</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>In Use Replica</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>In Use Replica</em>' attribute.
	 * @see #setInUseReplica(double)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getComponent_InUseReplica()
	 * @model
	 * @generated
	 */
	double getInUseReplica();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Component#getInUseReplica <em>In Use Replica</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>In Use Replica</em>' attribute.
	 * @see #getInUseReplica()
	 * @generated
	 */
	void setInUseReplica(double value);

	/**
	 * Returns the value of the '<em><b>Health Status</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Health Status</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Health Status</em>' attribute.
	 * @see #setHealthStatus(double)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getComponent_HealthStatus()
	 * @model
	 * @generated
	 */
	double getHealthStatus();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Component#getHealthStatus <em>Health Status</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Health Status</em>' attribute.
	 * @see #getHealthStatus()
	 * @generated
	 */
	void setHealthStatus(double value);

	/**
	 * Returns the value of the '<em><b>Performance Old</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Performance Old</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Performance Old</em>' attribute.
	 * @see #setPerformanceOld(double)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getComponent_PerformanceOld()
	 * @model
	 * @generated
	 */
	double getPerformanceOld();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Component#getPerformanceOld <em>Performance Old</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Performance Old</em>' attribute.
	 * @see #getPerformanceOld()
	 * @generated
	 */
	void setPerformanceOld(double value);

	/**
	 * Returns the value of the '<em><b>Original Request</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Original Request</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Original Request</em>' attribute.
	 * @see #setOriginalRequest(double)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getComponent_OriginalRequest()
	 * @model
	 * @generated
	 */
	double getOriginalRequest();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Component#getOriginalRequest <em>Original Request</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Original Request</em>' attribute.
	 * @see #getOriginalRequest()
	 * @generated
	 */
	void setOriginalRequest(double value);

	/**
	 * Returns the value of the '<em><b>Deployed On</b></em>' reference.
	 * It is bidirectional and its opposite is '{@link de.mdelab.morisia.comparch.Server#getDeployedComponent <em>Deployed Component</em>}'.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Deployed On</em>' reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Deployed On</em>' reference.
	 * @see #setDeployedOn(Server)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getComponent_DeployedOn()
	 * @see de.mdelab.morisia.comparch.Server#getDeployedComponent
	 * @model opposite="deployedComponent" required="true"
	 * @generated
	 */
	Server getDeployedOn();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Component#getDeployedOn <em>Deployed On</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Deployed On</em>' reference.
	 * @see #getDeployedOn()
	 * @generated
	 */
	void setDeployedOn(Server value);

	/**
	 * Returns the value of the '<em><b>Replica</b></em>' containment reference.
	 * It is bidirectional and its opposite is '{@link de.mdelab.morisia.comparch.Replica#getComponent <em>Component</em>}'.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Replica</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Replica</em>' containment reference.
	 * @see #setReplica(Replica)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getComponent_Replica()
	 * @see de.mdelab.morisia.comparch.Replica#getComponent
	 * @model opposite="component" containment="true"
	 * @generated
	 */
	Replica getReplica();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Component#getReplica <em>Replica</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Replica</em>' containment reference.
	 * @see #getReplica()
	 * @generated
	 */
	void setReplica(Replica value);

} // Component
