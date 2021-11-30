/**
 */
package de.mdelab.morisia.comparch;

import org.eclipse.emf.common.util.EList;

import org.eclipse.emf.ecore.EObject;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Cloud</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * </p>
 * <ul>
 *   <li>{@link de.mdelab.morisia.comparch.Cloud#getServers <em>Servers</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Cloud#getArchitecture <em>Architecture</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Cloud#getName <em>Name</em>}</li>
 * </ul>
 *
 * @see de.mdelab.morisia.comparch.ComparchPackage#getCloud()
 * @model
 * @generated
 */
public interface Cloud extends ArchitecturalElement {
	/**
	 * Returns the value of the '<em><b>Servers</b></em>' containment reference list.
	 * The list contents are of type {@link de.mdelab.morisia.comparch.Server}.
	 * It is bidirectional and its opposite is '{@link de.mdelab.morisia.comparch.Server#getCloud <em>Cloud</em>}'.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Servers</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Servers</em>' containment reference list.
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getCloud_Servers()
	 * @see de.mdelab.morisia.comparch.Server#getCloud
	 * @model opposite="cloud" containment="true" required="true"
	 * @generated
	 */
	EList<Server> getServers();

	/**
	 * Returns the value of the '<em><b>Architecture</b></em>' container reference.
	 * It is bidirectional and its opposite is '{@link de.mdelab.morisia.comparch.Architecture#getCloud <em>Cloud</em>}'.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Architecture</em>' reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Architecture</em>' container reference.
	 * @see #setArchitecture(Architecture)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getCloud_Architecture()
	 * @see de.mdelab.morisia.comparch.Architecture#getCloud
	 * @model opposite="cloud" required="true" transient="false"
	 * @generated
	 */
	Architecture getArchitecture();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Cloud#getArchitecture <em>Architecture</em>}' container reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Architecture</em>' container reference.
	 * @see #getArchitecture()
	 * @generated
	 */
	void setArchitecture(Architecture value);

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
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getCloud_Name()
	 * @model
	 * @generated
	 */
	String getName();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Cloud#getName <em>Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Name</em>' attribute.
	 * @see #getName()
	 * @generated
	 */
	void setName(String value);

} // Cloud
