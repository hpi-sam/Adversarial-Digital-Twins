/**
 */
package de.mdelab.morisia.comparch;

import org.eclipse.emf.common.util.EList;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Tenant</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * </p>
 * <ul>
 *   <li>{@link de.mdelab.morisia.comparch.Tenant#getName <em>Name</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Tenant#getArchitecture <em>Architecture</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Tenant#getComponents <em>Components</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Tenant#getReplicaFactor <em>Replica Factor</em>}</li>
 * </ul>
 *
 * @see de.mdelab.morisia.comparch.ComparchPackage#getTenant()
 * @model
 * @generated
 */
public interface Tenant extends ArchitecturalElement {
	/**
	 * Returns the value of the '<em><b>Name</b></em>' attribute.
	 * The default value is <code>""</code>.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Name</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Name</em>' attribute.
	 * @see #setName(String)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getTenant_Name()
	 * @model default=""
	 * @generated
	 */
	String getName();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Tenant#getName <em>Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Name</em>' attribute.
	 * @see #getName()
	 * @generated
	 */
	void setName(String value);

	/**
	 * Returns the value of the '<em><b>Architecture</b></em>' container reference.
	 * It is bidirectional and its opposite is '{@link de.mdelab.morisia.comparch.Architecture#getTenants <em>Tenants</em>}'.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Architecture</em>' container reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Architecture</em>' container reference.
	 * @see #setArchitecture(Architecture)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getTenant_Architecture()
	 * @see de.mdelab.morisia.comparch.Architecture#getTenants
	 * @model opposite="tenants" required="true" transient="false"
	 * @generated
	 */
	Architecture getArchitecture();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Tenant#getArchitecture <em>Architecture</em>}' container reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Architecture</em>' container reference.
	 * @see #getArchitecture()
	 * @generated
	 */
	void setArchitecture(Architecture value);

	/**
	 * Returns the value of the '<em><b>Components</b></em>' containment reference list.
	 * The list contents are of type {@link de.mdelab.morisia.comparch.Component}.
	 * It is bidirectional and its opposite is '{@link de.mdelab.morisia.comparch.Component#getTenant <em>Tenant</em>}'.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Components</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Components</em>' containment reference list.
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getTenant_Components()
	 * @see de.mdelab.morisia.comparch.Component#getTenant
	 * @model opposite="tenant" containment="true"
	 * @generated
	 */
	EList<Component> getComponents();

	/**
	 * Returns the value of the '<em><b>Replica Factor</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Replica Factor</em>' attribute isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Replica Factor</em>' attribute.
	 * @see #setReplicaFactor(double)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getTenant_ReplicaFactor()
	 * @model
	 * @generated
	 */
	double getReplicaFactor();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Tenant#getReplicaFactor <em>Replica Factor</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Replica Factor</em>' attribute.
	 * @see #getReplicaFactor()
	 * @generated
	 */
	void setReplicaFactor(double value);

} // Tenant
