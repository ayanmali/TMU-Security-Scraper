# Frontend Dockerfile
FROM node:20-alpine

# Set working directory
WORKDIR /app

ENV PATH /app/node_modules/.bin:$PATH

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy code files
COPY . ./

# Build the application
RUN npm run build

# Production stage
# FROM nginx:alpine

# Copy build files to nginx
# COPY --from=build /app/build /usr/share/nginx/html

# Copy custom nginx configuration (optional)
# COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port
# EXPOSE 3000

CMD ["npm", "start"]
