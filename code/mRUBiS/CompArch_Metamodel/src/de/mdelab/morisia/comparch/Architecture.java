/**
 */
package de.mdelab.morisia.comparch;

import org.eclipse.emf.common.util.EList;

/**
 * <!-- begin-user-doc -->
 * A representation of the model object '<em><b>Architecture</b></em>'.
 * <!-- end-user-doc -->
 *
 * <p>
 * The following features are supported:
 * </p>
 * <ul>
 *   <li>{@link de.mdelab.morisia.comparch.Architecture#getName <em>Name</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Architecture#getComponentTypes <em>Component Types</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Architecture#getInterfaceTypes <em>Interface Types</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Architecture#getTenants <em>Tenants</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Architecture#getCloud <em>Cloud</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.Architecture#getAnnotations <em>Annotations</em>}</li>
 * </ul>
 *
 * @see de.mdelab.morisia.comparch.ComparchPackage#getArchitecture()
 * @model
 * @generated
 */
public interface Architecture extends ArchitecturalElement {
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
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getArchitecture_Name()
	 * @model
	 * @generated
	 */
	String getName();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Architecture#getName <em>Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Name</em>' attribute.
	 * @see #getName()
	 * @generated
	 */
	void setName(String value);

	/**
	 * Returns the value of the '<em><b>Component Types</b></em>' containment reference list.
	 * The list contents are of type {@link de.mdelab.morisia.comparch.ComponentType}.
	 * It is bidirectional and its opposite is '{@link de.mdelab.morisia.comparch.ComponentType#getArchitecture <em>Architecture</em>}'.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Component Types</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Component Types</em>' containment reference list.
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getArchitecture_ComponentTypes()
	 * @see de.mdelab.morisia.comparch.ComponentType#getArchitecture
	 * @model opposite="architecture" containment="true"
	 * @generated
	 */
	EList<ComponentType> getComponentTypes();

	/**
	 * Returns the value of the '<em><b>Interface Types</b></em>' containment reference list.
	 * The list contents are of type {@link de.mdelab.morisia.comparch.InterfaceType}.
	 * It is bidirectional and its opposite is '{@link de.mdelab.morisia.comparch.InterfaceType#getArchitecture <em>Architecture</em>}'.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Interface Types</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Interface Types</em>' containment reference list.
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getArchitecture_InterfaceTypes()
	 * @see de.mdelab.morisia.comparch.InterfaceType#getArchitecture
	 * @model opposite="architecture" containment="true"
	 * @generated
	 */
	EList<InterfaceType> getInterfaceTypes();

	/**
	 * Returns the value of the '<em><b>Tenants</b></em>' containment reference list.
	 * The list contents are of type {@link de.mdelab.morisia.comparch.Tenant}.
	 * It is bidirectional and its opposite is '{@link de.mdelab.morisia.comparch.Tenant#getArchitecture <em>Architecture</em>}'.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Tenants</em>' containment reference list isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Tenants</em>' containment reference list.
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getArchitecture_Tenants()
	 * @see de.mdelab.morisia.comparch.Tenant#getArchitecture
	 * @model opposite="architecture" containment="true"
	 * @generated
	 */
	EList<Tenant> getTenants();

	/**
	 * Returns the value of the '<em><b>Cloud</b></em>' containment reference.
	 * It is bidirectional and its opposite is '{@link de.mdelab.morisia.comparch.Cloud#getArchitecture <em>Architecture</em>}'.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Cloud</em>' reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Cloud</em>' containment reference.
	 * @see #setCloud(Cloud)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getArchitecture_Cloud()
	 * @see de.mdelab.morisia.comparch.Cloud#getArchitecture
	 * @model opposite="architecture" containment="true" required="true"
	 * @generated
	 */
	Cloud getCloud();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Architecture#getCloud <em>Cloud</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Cloud</em>' containment reference.
	 * @see #getCloud()
	 * @generated
	 */
	void setCloud(Cloud value);

	/**
	 * Returns the value of the '<em><b>Annotations</b></em>' containment reference.
	 * It is bidirectional and its opposite is '{@link de.mdelab.morisia.comparch.Annotations#getArchitecture <em>Architecture</em>}'.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of the '<em>Annotations</em>' containment reference isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @return the value of the '<em>Annotations</em>' containment reference.
	 * @see #setAnnotations(Annotations)
	 * @see de.mdelab.morisia.comparch.ComparchPackage#getArchitecture_Annotations()
	 * @see de.mdelab.morisia.comparch.Annotations#getArchitecture
	 * @model opposite="architecture" containment="true"
	 * @generated
	 */
	Annotations getAnnotations();

	/**
	 * Sets the value of the '{@link de.mdelab.morisia.comparch.Architecture#getAnnotations <em>Annotations</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the new value of the '<em>Annotations</em>' containment reference.
	 * @see #getAnnotations()
	 * @generated
	 */
	void setAnnotations(Annotations value);

} // Architecture
