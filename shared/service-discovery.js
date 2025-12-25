/**
 * Service Discovery Module - Consul Integration
 * AquaWatch Project
 * 
 * Usage:
 *   const { registerService, discoverService, deregisterService } = require('./service-discovery');
 *   await registerService('my-service', 3000, '/health');
 */

const Consul = require('consul');

const CONSUL_HOST = process.env.CONSUL_HOST || 'consul';
const CONSUL_PORT = parseInt(process.env.CONSUL_PORT || '8500');

let consul = null;
let registeredServiceId = null;

/**
 * Get or create Consul client
 */
function getConsul() {
    if (!consul) {
        consul = new Consul({ host: CONSUL_HOST, port: CONSUL_PORT });
    }
    return consul;
}

/**
 * Register a service with Consul
 * @param {string} serviceName - Name of the service
 * @param {number} port - Port the service listens on (0 if no HTTP endpoint)
 * @param {string} healthPath - Health check endpoint path (e.g., '/health') - currently not used
 * @returns {Promise<string>} - Service ID
 */
async function registerService(serviceName, port = 0, healthPath = null) {
    const client = getConsul();
    const serviceId = `${serviceName}-${process.pid}`;

    // Simple registration without health checks for maximum compatibility
    const registration = {
        id: serviceId,
        name: serviceName,
        address: serviceName, // Docker service name
        port: port,
    };

    try {
        await client.agent.service.register(registration);
        registeredServiceId = serviceId;
        console.log(`✅ Service "${serviceName}" registered in Consul (ID: ${serviceId})`);
        return serviceId;
    } catch (error) {
        console.error(`❌ Failed to register service in Consul: ${error.message}`);
        throw error;
    }
}

/**
 * Discover a service by name
 * @param {string} serviceName - Name of the service to discover
 * @returns {Promise<{host: string, port: number}>} - Service address
 */
async function discoverService(serviceName) {
    const client = getConsul();

    try {
        const services = await client.catalog.service.nodes(serviceName);

        if (!services || services.length === 0) {
            throw new Error(`Service "${serviceName}" not found in Consul`);
        }

        // Return the first healthy service
        const service = services[0];
        return {
            host: service.ServiceAddress || service.Address,
            port: service.ServicePort
        };
    } catch (error) {
        console.error(`❌ Failed to discover service "${serviceName}": ${error.message}`);
        throw error;
    }
}

/**
 * Get all instances of a service
 * @param {string} serviceName - Name of the service
 * @returns {Promise<Array<{host: string, port: number}>>} - List of service instances
 */
async function discoverAllInstances(serviceName) {
    const client = getConsul();

    try {
        const services = await client.catalog.service.nodes(serviceName);
        return services.map(s => ({
            host: s.ServiceAddress || s.Address,
            port: s.ServicePort
        }));
    } catch (error) {
        console.error(`❌ Failed to discover service instances: ${error.message}`);
        return [];
    }
}

/**
 * Deregister the current service from Consul
 */
async function deregisterService() {
    if (!registeredServiceId) {
        return;
    }

    const client = getConsul();

    try {
        await client.agent.service.deregister(registeredServiceId);
        console.log(`✅ Service deregistered from Consul (ID: ${registeredServiceId})`);
        registeredServiceId = null;
    } catch (error) {
        console.error(`❌ Failed to deregister service: ${error.message}`);
    }
}

/**
 * Check if Consul is available
 * @returns {Promise<boolean>}
 */
async function isConsulAvailable() {
    const client = getConsul();

    try {
        await client.agent.members();
        return true;
    } catch (error) {
        return false;
    }
}

/**
 * Wait for Consul to be available
 * @param {number} maxRetries - Maximum number of retries
 * @param {number} delayMs - Delay between retries in milliseconds
 */
async function waitForConsul(maxRetries = 30, delayMs = 2000) {
    for (let i = 0; i < maxRetries; i++) {
        if (await isConsulAvailable()) {
            console.log('✅ Consul is available');
            return true;
        }
        console.log(`⏳ Waiting for Consul... (${i + 1}/${maxRetries})`);
        await new Promise(resolve => setTimeout(resolve, delayMs));
    }
    console.error('❌ Consul not available after retries');
    return false;
}

// Graceful shutdown
process.on('SIGTERM', deregisterService);
process.on('SIGINT', deregisterService);

module.exports = {
    registerService,
    discoverService,
    discoverAllInstances,
    deregisterService,
    isConsulAvailable,
    waitForConsul,
    getConsul
};
