/**
 * Eureka Service Discovery Client for Node.js microservices
 * Registers services with Netflix Eureka Server
 */

const Eureka = require('eureka-js-client').Eureka;

const EUREKA_HOST = process.env.EUREKA_HOST || 'eureka';
const EUREKA_PORT = parseInt(process.env.EUREKA_PORT || '8761');

let eurekaClient = null;

/**
 * Register a service with Eureka
 * @param {string} serviceName - Name of the service
 * @param {number} port - Port the service listens on
 * @param {string} healthPath - Health check endpoint path (e.g., '/health')
 * @returns {Promise<void>}
 */
async function registerService(serviceName, port, healthPath = '/health') {
    const hostname = serviceName;
    const ipAddr = serviceName; // Docker service name resolves to container IP

    const config = {
        instance: {
            app: serviceName.toUpperCase(),
            hostName: hostname,
            ipAddr: ipAddr,
            port: {
                '$': port,
                '@enabled': true
            },
            vipAddress: serviceName,
            dataCenterInfo: {
                '@class': 'com.netflix.appinfo.InstanceInfo$DefaultDataCenterInfo',
                name: 'MyOwn'
            },
            statusPageUrl: `http://${hostname}:${port}${healthPath}`,
            healthCheckUrl: `http://${hostname}:${port}${healthPath}`,
            homePageUrl: `http://${hostname}:${port}/`
        },
        eureka: {
            host: EUREKA_HOST,
            port: EUREKA_PORT,
            servicePath: '/eureka/apps/',
            maxRetries: 10,
            requestRetryDelay: 2000
        }
    };

    eurekaClient = new Eureka(config);

    return new Promise((resolve, reject) => {
        eurekaClient.start((error) => {
            if (error) {
                console.error(`❌ Failed to register with Eureka: ${error.message}`);
                reject(error);
            } else {
                console.log(`✅ Service "${serviceName}" registered with Eureka`);
                console.log(`📋 Health check: http://${hostname}:${port}${healthPath}`);
                resolve();
            }
        });
    });
}

/**
 * Discover a service by name
 * @param {string} serviceName - Name of the service to discover
 * @returns {{host: string, port: number} | null}
 */
function discoverService(serviceName) {
    if (!eurekaClient) {
        console.error('❌ Eureka client not initialized');
        return null;
    }

    try {
        const instances = eurekaClient.getInstancesByAppId(serviceName.toUpperCase());
        if (instances && instances.length > 0) {
            const instance = instances[0];
            return {
                host: instance.hostName,
                port: instance.port['$']
            };
        }
    } catch (error) {
        console.error(`❌ Failed to discover service ${serviceName}: ${error.message}`);
    }
    return null;
}

/**
 * Wait for Eureka to be available
 * @param {number} maxRetries - Maximum number of retries
 * @param {number} retryDelay - Delay between retries in ms
 * @returns {Promise<boolean>}
 */
async function waitForEureka(maxRetries = 30, retryDelay = 2000) {
    const http = require('http');

    for (let i = 0; i < maxRetries; i++) {
        try {
            await new Promise((resolve, reject) => {
                const req = http.get(`http://${EUREKA_HOST}:${EUREKA_PORT}/eureka/apps`, (res) => {
                    if (res.statusCode === 200) {
                        resolve(true);
                    } else {
                        reject(new Error(`Status ${res.statusCode}`));
                    }
                });
                req.on('error', reject);
                req.setTimeout(5000, () => reject(new Error('Timeout')));
            });
            console.log('✅ Eureka Server is available');
            return true;
        } catch (error) {
            console.log(`⏳ Waiting for Eureka... (${i + 1}/${maxRetries})`);
            await new Promise(r => setTimeout(r, retryDelay));
        }
    }
    console.log('⚠️ Eureka Server not available after waiting');
    return false;
}

/**
 * Graceful shutdown - deregister from Eureka
 */
function setupGracefulShutdown() {
    const shutdown = () => {
        if (eurekaClient) {
            console.log('🛑 Deregistering from Eureka...');
            eurekaClient.stop();
        }
    };

    process.on('SIGTERM', shutdown);
    process.on('SIGINT', shutdown);
}

module.exports = {
    registerService,
    discoverService,
    waitForEureka,
    setupGracefulShutdown
};
